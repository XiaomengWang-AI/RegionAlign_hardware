"""Frozen clean-only final benchmarks, independent of fault/search runtimes.

Prepare hashes without deserializing final examples, preflight on synthetic
inputs, then run each frozen condition with ``--split test --final``. TextVQA
public validation is treated as final/test here. No partial final evaluation,
fault selection, gradients, permutation, or resume with changed inputs exists.

WikiText joins all Arrow rows with two newlines, tokenizes without special
tokens, and scores every token except the first exactly once (512 context,
256 stride). Each window contributes a token-summed cross entropy; shared
context is never counted twice. VQA uses image plus question without OCR hints,
greedy decoding capped at 32 new tokens, and the vendored official-consensus
scorer. Answer NLL uses the most frequent normalized reference, lexicographic
ties, without EOS. If the selected normalized answer is empty, use the most
frequent nonempty raw answer in that group (same tie rule). Entirely blank
references have no answer tokens: generation is still scored, and unavailable
NLL is counted separately. LLaVA uses its USER/ASSISTANT template; Qwen uses its saved
chat template and 256..1280 image tokens. These choices are frozen before final
examples are opened. Vision/projector parameters retain checkpoint dtype;
only the full language backbone is quantized using the existing schema 2.

Every condition is a new process with a fresh model and output directory.
Failures stay visible; missing/nonfinite records prevent successful metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

from regionalign.experiments import clean_mmlu as M
from regionalign import config as C

SOURCES = (
    "regionalign/__init__.py", "regionalign/experiments/__init__.py",
    "regionalign/harness/__init__.py", "regionalign/runtimes/__init__.py",
    "regionalign/experiments/clean_benchmarks.py", "regionalign/experiments/clean_mmlu.py",
    "regionalign/config.py", "regionalign/harness/core.py",
    "regionalign/runtime.json", "regionalign/runtimes/vqa_score.py",
)
VLM_TYPES = {"llava-1.5-7b": "llava", "qwen2-vl-7b": "qwen2_vl"}
SETTINGS = {
    "context": 512, "stride": 256, "join": "two_newlines", "add_special_tokens": False,
    "max_new_tokens": 32, "do_sample": False, "batch_size": 1,
    "qwen_min_pixels": 256 * 28 * 28, "qwen_max_pixels": 1280 * 28 * 28,
    "answer_reference": "most_frequent_normalized_lexicographic_tie_raw_fallback_no_eos",
    "empty_reference_nll": "unavailable_zero_target_tokens",
    "ocr_hints": False, "int4_group": 128, "min_free_gpu_gib": 40,
}


def windows(length, context=512, stride=256):
    """Yield (input_start, input_end, first_scored_token) with exact coverage."""
    if length < 2 or not 1 <= stride < context:
        raise ValueError("need at least two tokens and 1 <= stride < context")
    previous = 1
    for start in range(0, length - 1, stride):
        end = min(start + context, length)
        first = max(previous, start + 1)
        if first < end:
            yield start, end, first
        previous = end
        if end == length:
            break


def require_final(split, final):
    if split != "test" or not final:
        raise ValueError("final benchmark requires --split test --final")


def source_records():
    return [M.file_record(C.PROJECT / name, C.PROJECT) for name in SOURCES]


def prepare(args):
    audit = M.read_json(args.cache_audit)
    models = {}
    for key in sorted(set(M.PAPER_MODELS) | set(VLM_TYPES)):     # the frozen final matrix
        path = Path(audit["models"][key]["path"])
        if key in M.MODEL_SPECS:
            models[key] = M.checkpoint_manifest(path, key)
        else:
            config = M.read_json(path / "config.json")
            if config["model_type"] != VLM_TYPES[key]:
                raise ValueError(f"wrong model type for {key}")
            files = set(M.checkpoint_files(path)) | set(path.glob("*.json"))
            models[key] = {"path": str(path), "model_type": config["model_type"],
                           "files": [M.file_record(p, path) for p in sorted(files)]}
        print("checkpoint frozen", key, flush=True)
    dataset_specs = {
        "wikitext": ("wikitext-2-raw", "wikitext-test.arrow", None),
        "okvqa": ("ok-vqa", "ok-vqa-val2014-", 5046),
        "textvqa": ("text-vqa", "textvqa-validation-", 5000),
    }
    datasets = {}
    for task, (key, pattern, count) in dataset_specs.items():
        files = [Path(p) for p in audit["datasets"][key]["files"] if pattern in Path(p).name]
        if not files:
            raise ValueError(f"missing final input for {task}")
        datasets[task] = {"files": [M.file_record(p) for p in sorted(files)], "expected_count": count}
    conditions = [{"model": model, "format": fmt, "task": task}
                  for model in sorted(models) for fmt in ("int8", "int4")
                  for task in (("wikitext",) if model in M.MODEL_SPECS else ("okvqa", "textvqa"))]
    freeze_plan(args.output, models, datasets, conditions)


def freeze_plan(output, models, datasets, conditions, **metadata):
    run = C.fresh_output_dir("clean_benchmarks", "final", output)
    (run / "source").mkdir()
    sources = source_records()
    for record in sources:
        target = run / "source" / record["path"].replace("/", "__")
        shutil.copyfile(C.PROJECT / record["path"], target)
        if M.sha256(target) != record["sha256"]:
            raise RuntimeError("source changed during freezing")
    plan = {"scope": "clean_benchmarks_v1", "created_at": M.utc_now(),
            "models": models, "datasets": datasets, "conditions": conditions,
            "settings": SETTINGS, "sources": sources, "no_faults_or_search": True,
            **metadata}
    M.write_json_once(run / "plan.json", plan)
    M.write_json_once(run / "test_selection.json", {"plan_sha256": M.sha256(run / "plan.json"),
                                                   "conditions": conditions})
    print(run, flush=True)


def prepare_retry(args):
    """Freeze failed clean VLM conditions, retaining prior final-access evidence.

    Inherit exact input hashes, which execution rechecks before reading examples.
    Historical source copies must still match their hashes; the live evaluator
    may differ because the new plan records an explicit implementation repair.
    Completed conditions are referenced, never resubmitted or overwritten.
    """
    parent = Path(args.parent_run).resolve()
    old = M.read_json(parent / "plan.json")
    parent_sha = M.sha256(parent / "plan.json")
    if old.get("scope") != "clean_benchmarks_v1" or old.get("no_faults_or_search") is not True:
        raise ValueError("retry requires a frozen clean-only parent")
    if M.read_json(parent / "test_selection.json") != {
        "plan_sha256": parent_sha, "conditions": old["conditions"]
    }:
        raise ValueError("parent selection changed")
    if {r["path"] for r in old["sources"]} != set(SOURCES):
        raise ValueError("parent source inventory changed")
    for r in old["sources"]:
        if M.sha256(parent / "source" / r["path"].replace("/", "__")) != r["sha256"]:
            raise ValueError("parent source snapshot changed")
    allowed = {"answer_reference", "empty_reference_nll"}
    if {k: v for k, v in old["settings"].items() if k not in allowed} != {
        k: v for k, v in SETTINGS.items() if k not in allowed
    }:
        raise ValueError("retry cannot change other benchmark settings")
    conditions, failures, reused = [], [], []
    for condition in old["conditions"]:
        target = parent / "conditions" / condition["model"] / condition["format"] / condition["task"]
        metrics, failure = target / "metrics.json", target / "failure.json"
        if metrics.exists():
            if failure.exists() or M.read_json(metrics).get("status") != "complete":
                raise ValueError("inconsistent parent completion state")
            reused.append(M.file_record(metrics))
        elif failure.exists():
            if condition["model"] not in VLM_TYPES or condition["task"] not in {"okvqa", "textvqa"}:
                raise ValueError("this retry only repairs failed VLM baselines")
            error = M.read_json(failure)
            if error.get("type") != "ValueError" or error.get("error") != "empty VQA references":
                raise ValueError("unrecognized failure requires separate diagnosis")
            conditions.append(condition)
            failures.append(M.file_record(failure))
        else:
            raise ValueError("parent has unfinished conditions; check running jobs first")
    if not conditions:
        raise ValueError("no failed clean VLM conditions to retry")
    freeze_plan(args.output,
                {k: v for k, v in old["models"].items() if k in {c["model"] for c in conditions}},
                {k: v for k, v in old["datasets"].items() if k in {c["task"] for c in conditions}},
                conditions, retry={"parent_run": str(parent), "parent_plan_sha256": parent_sha,
                    "prior_final_access": M.file_record(parent / "test_access.jsonl"),
                    "failures": failures, "reused_completed_metrics": reused,
                    "reason": "repair empty normalized VQA reference handling",
                    "prior_final_exposure": True})


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if plan.get("scope") != "clean_benchmarks_v1" or plan.get("settings") != SETTINGS:
        raise ValueError("wrong clean benchmark protocol")
    selection = M.read_json(run / "test_selection.json")
    if selection != {"plan_sha256": M.sha256(run / "plan.json"), "conditions": plan["conditions"]}:
        raise ValueError("final selection changed")
    M.verify_records(C.PROJECT, plan["sources"])
    for record in plan["sources"]:
        if M.sha256(run / "source" / record["path"].replace("/", "__")) != record["sha256"]:
            raise ValueError("source snapshot changed")
    return plan


def rows_from_arrow(records):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    for record in records:
        with pa.memory_map(record["path"], "r") as handle:
            for batch in ipc.open_stream(handle):
                for i in range(batch.num_rows):
                    yield batch.slice(i, 1).to_pylist()[0]


def verify_dataset_files(records):
    """Retain absolute dataset identities while using the relative-path verifier."""
    relative = []
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("dataset identity must be an absolute normalized path")
        relative.append({**record, "path": str(path.relative_to("/"))})
    M.verify_records("/", relative)


def load_clean(model_key, checkpoint, fmt, device):
    import torch
    if model_key in M.MODEL_SPECS:
        model, processor, audit, timing = M.load_model(checkpoint, fmt, device)
        return model, processor, model, audit, timing
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration
    from regionalign.harness import core as P
    start = time.monotonic()
    free, _ = torch.cuda.mem_get_info(device)
    if free < SETTINGS["min_free_gpu_gib"] * 1024 ** 3:
        raise RuntimeError("clean VLM evaluation requires 40 GiB free GPU memory")
    cls = LlavaForConditionalGeneration if model_key == "llava-1.5-7b" else Qwen2VLForConditionalGeneration
    model = cls.from_pretrained(checkpoint, local_files_only=True, torch_dtype="auto",
                                device_map={"": device}, attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    extra = {} if model_key == "llava-1.5-7b" else {
        "min_pixels": SETTINGS["qwen_min_pixels"], "max_pixels": SETTINGS["qwen_max_pixels"]}
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, **extra)
    if model_key == "llava-1.5-7b":
        language = model.language_model
    else:
        language = torch.nn.Module()
        language.add_module("model", model.model)
        language.add_module("lm_head", model.lm_head)
    language_ids = {id(p) for p in language.parameters()}
    excluded = [(n, p) for n, p in model.named_parameters() if id(p) not in language_ids]
    before = [M._tensor_record(n, p) for n, p in excluded]
    loaded = time.monotonic()
    P.quantize(language, fmt)
    audit = P.quantization_audit(language)
    after = [M._tensor_record(n, p) for n, p in excluded]
    if before != after or audit["quantized_fraction"] != 1.0:
        raise RuntimeError("language coverage or unchanged visual-parameter check failed")
    audit = {**audit, "visual_parameters": after, "visual_parameters_unchanged": True,
             "language_parameter_names": [n for n, p in model.named_parameters() if id(p) in language_ids]}
    return model, processor, language, audit, {"load_seconds": loaded - start,
                                             "quantize_seconds": time.monotonic() - loaded}


def score_wikitext(model, tokenizer, text, device, record):
    import torch
    import torch.nn.functional as F
    ids = tokenizer.encode(text, add_special_tokens=False)
    total = 0.0
    count = 0
    for i, (start, end, first) in enumerate(windows(len(ids), SETTINGS["context"], SETTINGS["stride"])):
        inputs = torch.tensor([ids[start:end]], device=device)
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False).logits
            scores = logits[0, first-start-1:end-start-1].float()
            targets = inputs[0, first-start:]
            loss = float(F.cross_entropy(scores, targets, reduction="none").double().sum().item())
        finite = math.isfinite(loss)
        record({"window": i, "start": start, "end": end, "first_target": first,
                "target_tokens": end-first, "nll_sum": loss if finite else None,
                "status": "complete" if finite else "nonfinite"})
        if not finite:
            raise RuntimeError("nonfinite WikiText loss")
        total += loss
        count += end-first
    if count != len(ids)-1:
        raise RuntimeError("WikiText target coverage mismatch")
    return {"target_tokens": count, "nll_sum": total, "nll": total/count,
            "ppl": math.exp(total/count) if total/count < 709 else None,
            "input_ids_sha256": M.object_sha256(ids), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def vlm_inputs(key, processor, image, question, device):
    text = question + "\nAnswer the question using a single word or phrase."
    if key == "llava-1.5-7b":
        prompt = "USER: <image>\n" + text + " ASSISTANT:"
    else:
        prompt = processor.apply_chat_template([
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ], tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=[image], return_tensors="pt").to(device)


def reference_answer(answers, task):
    from regionalign.runtimes.vqa_score import normalize_vqa_answer
    if not answers or not all(isinstance(x, str) for x in answers):
        raise ValueError("missing or non-string VQA references")
    normalized = [normalize_vqa_answer(x, textvqa=task == "textvqa") for x in answers]
    counts = Counter(normalized)
    selected = min(counts, key=lambda x: (-counts[x], x))
    if selected:
        return selected
    raw = Counter(x.strip() for x, norm in zip(answers, normalized) if not norm and x.strip())
    return min(raw, key=lambda x: (-raw[x], x)) if raw else ""


def score_vqa(model, processor, key, image, question, answers, task, device):
    import torch
    import torch.nn.functional as F
    from regionalign.runtimes.vqa_score import vqa_score
    inputs = vlm_inputs(key, processor, image, question, device)
    length = inputs.input_ids.shape[1]
    reference = reference_answer(answers, task)
    answer_ids = processor.tokenizer.encode(reference, add_special_tokens=False)
    if reference and not answer_ids:
        raise ValueError("reference has no answer tokens")
    nll_sum = 0.0
    with torch.inference_mode():
        if answer_ids:
            teacher = dict(inputs)
            suffix = torch.tensor([answer_ids], device=device)
            teacher["input_ids"] = torch.cat([inputs.input_ids, suffix], dim=1)
            teacher["attention_mask"] = torch.ones_like(teacher["input_ids"])
            output = model(**teacher, use_cache=False)
            scores = output.logits[0, -len(answer_ids)-1:-1].float()
            nll_sum = float(F.cross_entropy(scores, suffix[0], reduction="none").double().sum().item())
            del output, scores, teacher
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                   max_new_tokens=SETTINGS["max_new_tokens"], use_cache=True,
                                   return_dict_in_generate=True, output_logits=True)
    finite = math.isfinite(nll_sum) and all(bool(torch.isfinite(x).all()) for x in generated.logits)
    tokens = generated.sequences[0, length:].tolist()
    prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    truncated = len(tokens) == SETTINGS["max_new_tokens"] and (not tokens or tokens[-1] not in eos)
    return {"prediction": prediction, "score": vqa_score(prediction, answers, task=task),
            "target_tokens": len(answer_ids), "reference": reference,
            "nll_available": bool(answer_ids),
            "nll_unavailable_reason": None if answer_ids else "empty_reference",
            "nll_sum": nll_sum if math.isfinite(nll_sum) else None,
            "generated_tokens": len(tokens), "truncated": truncated,
            "input_ids_sha256": M.object_sha256(inputs.input_ids[0].tolist()),
            "status": "complete" if finite else "nonfinite"}


def run_condition(args):
    require_final(args.split, args.final)  # Before opening final data or creating output.
    C.setup_environment()
    run = Path(args.run).resolve()
    plan = load_plan(run)
    condition = {"model": args.model, "format": args.format, "task": args.task}
    if condition not in plan["conditions"]:
        raise ValueError("condition was not frozen")
    target = run / "conditions" / args.model / args.format / args.task
    target.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = plan["models"][args.model]
        dataset = plan["datasets"][args.task]
        M.verify_records(checkpoint["path"], checkpoint["files"])
        verify_dataset_files(dataset["files"])
        preflight_path = run / "preflight" / f"{args.model}_{args.format}.json"
        preflight = M.read_json(preflight_path)
        if preflight.get("status") != "complete" or preflight.get("plan_sha256") != M.sha256(run / "plan.json"):
            raise ValueError("missing successful synthetic preflight for this plan")
        import torch
        model, processor, language, audit, timing = load_clean(args.model, checkpoint["path"], args.format, args.device)
        M.write_json_once(target / "quantization_audit.json", audit)
        M.write_json_once(target / "quantized_state.json", M.quantized_state_manifest(language))
        M.write_json_once(target / "execution.json", {"condition": condition, "plan_sha256": M.sha256(run / "plan.json"),
            "environment": M.software_environment(), "device": M.runtime_device_identity(torch, args.device),
            "preflight_sha256": M.sha256(preflight_path), "created_at": M.utc_now()})
        access = {"scope": plan["scope"], "condition": condition, "time": M.utc_now(), "split": "test",
                  "final": True, "plan_sha256": M.sha256(run / "plan.json"), "purpose": "clean_only_final"}
        M.append_jsonl(run / "test_access.jsonl", access)
        M.append_jsonl(C.RESULTS_ROOT / "test_access.jsonl", {**access, "run": str(run)})
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(args.device)
        with (target / "samples.jsonl").open("x") as handle:
            def record(row):
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            if args.task == "wikitext":
                text = "\n\n".join(row["text"] for row in rows_from_arrow(dataset["files"]))
                metrics = score_wikitext(model, processor, text, args.device, record)
            else:
                from PIL import Image
                count = failed = truncated = tokens = unavailable_nll = 0
                score_sum = nll_sum = 0.0
                seen = set()
                for row in rows_from_arrow(dataset["files"]):
                    qid = str(row["question_id"])
                    if qid in seen:
                        raise ValueError("duplicate VQA question ID")
                    seen.add(qid)
                    raw_image = row["image"]
                    if not raw_image.get("bytes"):
                        raise ValueError("VQA input must embed image bytes in frozen Arrow")
                    with Image.open(io.BytesIO(raw_image["bytes"])) as source:
                        image = source.convert("RGB")
                    result = score_vqa(model, processor, args.model, image, row["question"],
                                       row["answers"], args.task, args.device)
                    result.update(index=count, question_id=qid,
                                  image_sha256=hashlib.sha256(raw_image["bytes"]).hexdigest())
                    record(result)
                    count += 1
                    failed += result["status"] != "complete"
                    truncated += result["truncated"]
                    score_sum += result["score"]
                    tokens += result["target_tokens"]
                    unavailable_nll += not result["nll_available"]
                    nll_sum += result["nll_sum"] or 0.0
                    if count % 100 == 0:
                        print(json.dumps({"completed": count, "expected": dataset["expected_count"]}), flush=True)
                if count != dataset["expected_count"] or failed:
                    raise RuntimeError(f"VQA incomplete/nonfinite: {count=}, {failed=}")
                metrics = {"count": count, "score": score_sum/count, "target_tokens": tokens,
                           "nll": nll_sum/tokens if tokens else None, "nll_sum": nll_sum,
                           "nll_unavailable_samples": unavailable_nll, "failed_samples": failed,
                           "truncated_samples": truncated, "truncation_rate": truncated/count}
        load_plan(run)  # Refuse publication after source drift during execution.
        metrics.update(status="complete", **condition, split="test", no_faults_or_search=True,
                       completed_at=M.utc_now(), samples_sha256=M.sha256(target / "samples.jsonl"),
                       timing={**timing, "evaluation_seconds": time.monotonic()-started},
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device))
        M.write_json_once(target / "metrics.json", metrics)
        print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    except Exception as error:
        M.write_json_once(target / "failure.json", {"status": "failed", "type": type(error).__name__,
                                                   "error": str(error), "time": M.utc_now()})
        raise


def preflight(args):
    """Model execution on generated text/image only; no benchmark rows opened."""
    C.setup_environment()
    run = Path(args.run).resolve()
    plan = load_plan(run)
    checkpoint = plan["models"][args.model]
    M.verify_records(checkpoint["path"], checkpoint["files"])
    model, processor, language, audit, timing = load_clean(args.model, checkpoint["path"], args.format, args.device)
    if args.model in M.MODEL_SPECS:
        result = score_wikitext(model, processor, "The cat is on the mat. " * 160, args.device, lambda _: None)
    else:
        from PIL import Image
        result = []
        for task in ("okvqa", "textvqa"):
            for answers in (["white"] * 10, ["white"] * 9 + ["a"], ["a"] * 10, [""] * 10):
                row = score_vqa(model, processor, args.model, Image.new("RGB", (336, 336), "white"),
                                "What color is the image?", answers, task, args.device)
                if row["status"] != "complete":
                    raise RuntimeError("nonfinite synthetic preflight")
                result.append({"task": task, "synthetic_answers": answers, **row})
    target = run / "preflight" / f"{args.model}_{args.format}.json"
    target.parent.mkdir(exist_ok=True)
    M.write_json_once(target, {"status": "complete", "scope": "synthetic_input_only",
                              "plan_sha256": M.sha256(run / "plan.json"), "result": result,
                              "audit": audit, "timing": timing, "completed_at": M.utc_now()})
    print(target, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache-audit", required=True)
    prep.add_argument("--output", required=True)
    prep.set_defaults(function=prepare)
    retry = sub.add_parser("prepare-retry")
    retry.add_argument("--parent-run", required=True)
    retry.add_argument("--output", required=True)
    retry.set_defaults(function=prepare_retry)
    for name, function in [("preflight", preflight), ("run", run_condition)]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", required=True)
        cmd.add_argument("--model", required=True, choices=sorted(set(M.PAPER_MODELS) | set(VLM_TYPES)))
        cmd.add_argument("--format", required=True, choices=("int8", "int4"))
        cmd.add_argument("--device", default="cuda:0")
        if name == "run":
            cmd.add_argument("--task", required=True, choices=("wikitext", "okvqa", "textvqa"))
            cmd.add_argument("--split", required=True, choices=("test",))
            cmd.add_argument("--final", action="store_true")
        cmd.set_defaults(function=function)
    args = p.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()

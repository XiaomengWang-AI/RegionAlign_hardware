"""Independent, clean-only WikiText validation evaluation for local LLMs.

Prepare one tokenizer-independent panel, then reuse it across all conditions::

    python -B -m regionalign.experiments.clean_wikitext prepare \
        --data-file /cache/wikitext-validation.arrow --output regionalign/results/clean/panel
    python -B -m regionalign.experiments.clean_wikitext run \
        --data-file /cache/wikitext-validation.arrow --panel regionalign/results/clean/panel/panel.json \
        --model-dir /checkpoint --model-name llama2-7b --format int8 \
        --device cuda:0 --output regionalign/results/clean/llama2_int8

Only validation Arrow input is accepted. This module has no search, gradient,
fault, permutation, or final-test evaluation path. The panel consists of fixed,
nonoverlapping raw-text spans selected before tokenization. Each model scores
the prefix of at most 512 tokens in each span. The first input token is context;
all remaining tokens are targets. NLL is weighted by target-token count, not
averaged across examples. This is a short-context development measurement,
not full-corpus WikiText test perplexity. Every output directory is new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

from regionalign.experiments import clean_mmlu as M
from regionalign import config as C


SCHEMA = 1
PANEL_COUNT = 128
WORDS_PER_SAMPLE = 128
SEED = 20261109
MAX_TOKENS = 512
DEFAULT_MIN_FREE_GPU_GIB = 8
SOURCES = (
    "regionalign/__init__.py", "regionalign/experiments/__init__.py",
    "regionalign/harness/__init__.py",
    "regionalign/experiments/clean_wikitext.py", "regionalign/experiments/clean_mmlu.py",
    "regionalign/config.py", "regionalign/harness/core.py",
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validation_path(raw):
    """Reject non-validation filenames before hashing or opening their contents."""
    path = Path(raw).expanduser().absolute()
    allowed = {"validation.arrow", "wikitext-validation.arrow"}
    if path.name not in allowed:
        raise ValueError("only WikiText validation.arrow input is permitted")
    resolved = path.resolve()
    if resolved.name not in allowed:
        raise ValueError("validation input must not resolve to a different split")
    if any(re.search(r"(^|[-_.])test($|[-_.])", part, re.I)
           for part in resolved.parts):
        raise ValueError("test paths are not permitted")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_validation_text(raw):
    path = validation_path(raw)
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    if "text" not in table.column_names:
        raise ValueError("WikiText Arrow input must contain a text column")
    texts = table["text"].to_pylist()
    if not texts or not all(isinstance(value, str) for value in texts):
        raise ValueError("WikiText text column must contain non-null strings")
    joined = "\n".join(texts)
    return joined, {
        "path": str(path), "sha256": M.sha256(path), "split": "validation",
        "rows": len(texts), "joined_text_sha256": text_sha256(joined),
        "join_rule": "newline_between_arrow_rows", "characters": len(joined),
    }


def select_spans(text, count=PANEL_COUNT, words_per_sample=WORDS_PER_SAMPLE, seed=SEED):
    """Choose disjoint full whitespace-word blocks, retaining source offsets."""
    if count < 1 or words_per_sample < 2:
        raise ValueError("count must be positive and words_per_sample at least two")
    words = list(re.finditer(r"\S+", text))
    candidates = len(words) // words_per_sample
    if candidates < count:
        raise ValueError(f"only {candidates} disjoint full blocks for requested {count}")
    chosen = sorted(random.Random(seed).sample(range(candidates), count))
    rows = []
    for index, block in enumerate(chosen):
        first = block * words_per_sample
        start, end = words[first].start(), words[first + words_per_sample - 1].end()
        snippet = text[start:end]
        rows.append({"index": index, "block": block, "char_start": start,
                     "char_end": end, "word_start": first,
                     "word_end": first + words_per_sample, "text": snippet,
                     "text_sha256": text_sha256(snippet)})
    return rows


def prepare(args):
    text, source = read_validation_text(args.data_file)
    rows = select_spans(text, args.count, args.words_per_sample, args.seed)
    output = C.fresh_output_dir("clean_evaluation", "wikitext_validation_panel", args.output)
    panel = {"schema_version": SCHEMA, "scope": "clean_wikitext_validation",
             "created_at": M.utc_now(), "source": source,
             "selection": {"count": args.count, "words_per_sample": args.words_per_sample,
                           "seed": args.seed, "method": "fixed_nonoverlapping_word_blocks"},
             "samples": rows}
    M.write_json_once(output / "panel.json", panel)
    M.write_json_once(output / "status.json", {"status": "complete", "count": len(rows),
                                               "panel_sha256": M.sha256(output / "panel.json")})
    print(output / "panel.json", flush=True)


def load_panel(panel_path, data_file):
    text, source = read_validation_text(data_file)
    panel_path = Path(panel_path)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("schema_version") != SCHEMA or panel.get("scope") != "clean_wikitext_validation":
        raise ValueError("unsupported validation panel")
    if panel.get("source") != source:
        raise ValueError("validation source identity or SHA256 differs from frozen panel")
    selection = panel["selection"]
    expected = select_spans(text, selection["count"], selection["words_per_sample"], selection["seed"])
    if panel.get("samples") != expected:
        raise ValueError("panel spans or hashes differ from the frozen sampling rule")
    return panel


def aggregate(rows, expected_count):
    """Do not report success or a finite-only aggregate if any sample failed."""
    failed = [row for row in rows if row.get("status") != "complete"]
    valid = [row for row in rows if row.get("status") == "complete"]
    for row in valid:
        count, value = row.get("target_tokens"), row.get("nll_sum")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError("invalid completed NLL record")
    complete = len(rows) == expected_count and not failed
    tokens = sum(row["target_tokens"] for row in valid)
    nll_sum = math.fsum(row["nll_sum"] for row in valid)
    nll = nll_sum / tokens if complete and tokens else None
    try:
        ppl = math.exp(nll) if nll is not None else None
    except OverflowError:
        ppl = None
    return {"status": "complete" if complete else "failed",
            "expected_samples": expected_count, "observed_samples": len(rows),
            "failed_samples": len(failed), "finite_target_tokens": tokens,
            "nll_sum": nll_sum if complete else None, "nll": nll, "ppl": ppl,
            "ppl_overflow": complete and nll is not None and ppl is None,
            "truncated_samples": sum(bool(row.get("truncated")) for row in rows)}


def score_sample(model, tokenizer, sample, device, max_tokens):
    import torch
    import torch.nn.functional as F

    full_ids = tokenizer.encode(sample["text"], add_special_tokens=True)
    ids = full_ids[:max_tokens]
    record = {"index": sample["index"], "text_sha256": sample["text_sha256"],
              "char_start": sample["char_start"], "char_end": sample["char_end"],
              "full_input_tokens": len(full_ids), "input_tokens": len(ids),
              "target_tokens": max(0, len(ids) - 1), "truncated": len(ids) < len(full_ids),
              "input_ids_sha256": M.object_sha256(ids)}
    if len(ids) < 2:
        return {**record, "status": "insufficient_tokens", "nll_sum": None, "nll": None}
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        logits = output.logits[:, :-1, :].float().contiguous()
        targets = inputs[:, 1:].contiguous()
        losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="none")
        nll_sum = float(losses.to(torch.float64).sum().item())
    finite = math.isfinite(nll_sum)
    return {**record, "status": "complete" if finite else "nonfinite",
            "nll_sum": nll_sum if finite else None,
            "nll": nll_sum / record["target_tokens"] if finite else None}


def run(args):
    # Validate data identity before allocating a fresh output or loading a model.
    panel = load_panel(args.panel, args.data_file)
    output = C.fresh_output_dir("clean_evaluation", "wikitext_validation", args.output)
    started = time.monotonic()
    try:
        C.setup_environment()
        checkpoint = M.checkpoint_manifest(args.model_dir, args.model_name)
        source_dir = output / "source"
        source_dir.mkdir()
        sources = []
        for relative in SOURCES:
            source = C.PROJECT / relative
            target = source_dir / (relative.replace("/", "__") + ".snapshot")
            shutil.copyfile(source, target)
            current_sha, copied_sha = M.sha256(source), M.sha256(target)
            if current_sha != copied_sha:
                raise RuntimeError(f"source changed while being frozen: {relative}")
            sources.append({"path": relative, "sha256": current_sha,
                            "snapshot_sha256": copied_sha})
        panel_file = output / "panel.json"
        M.write_json_once(panel_file, panel)
        import torch
        device = M.runtime_device_identity(torch, args.device)
        environment = M.software_environment()
        protocol = {"scope": "clean_wikitext_validation", "split": "validation",
                    "format": args.format, "model_name": args.model_name,
                    "max_input_tokens": args.max_tokens, "batch_size": 1,
                    "add_special_tokens": True, "use_cache": False,
                    "compute_dtype": "bfloat16", "sample_weighting": "target_tokens",
                    "min_free_gpu_gib": args.min_free_gpu_gib,
                    "context": "independent_prefix_per_frozen_raw_text_span",
                    "reference_panel_sha256": M.sha256(args.panel),
                    "copied_panel_sha256": M.sha256(panel_file),
                    "checkpoint": checkpoint, "sources": sources,
                    "device": device, "environment": environment,
                    "created_at": M.utc_now(), "no_faults_or_search": True}
        M.write_json_once(output / "protocol.json", protocol)
        min_free_gpu_bytes = int(args.min_free_gpu_gib * 1024 ** 3)
        model, tokenizer, audit, timings = M.load_model(
            str(args.model_dir), args.format, args.device,
            min_free_gpu_bytes=min_free_gpu_bytes)
        if args.max_tokens > model.config.max_position_embeddings:
            raise ValueError("requested input length exceeds model context")
        M.write_json_once(output / "quantization_audit.json", audit)
        M.write_json_once(output / "quantized_state.json", M.quantized_state_manifest(model))
        rows = []
        eval_started = time.monotonic()
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        with (output / "samples.jsonl").open("x", encoding="utf-8") as handle:
            for sample in panel["samples"]:
                record = score_sample(model, tokenizer, sample, args.device, args.max_tokens)
                rows.append(record)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
        result = aggregate(rows, len(panel["samples"]))
        result.update({"format": args.format, "model_name": args.model_name,
                       "protocol_sha256": M.sha256(output / "protocol.json"),
                       "samples_sha256": M.sha256(output / "samples.jsonl"),
                       "quantization_audit_sha256": M.sha256(output / "quantization_audit.json"),
                       "quantized_state_file_sha256": M.sha256(output / "quantized_state.json"),
                       "timing": {**timings, "evaluation_seconds": time.monotonic() - eval_started,
                                  "total_seconds": time.monotonic() - started},
                       "peak_evaluation_gpu_bytes": torch.cuda.max_memory_allocated(args.device)
                           if str(args.device).startswith("cuda") else None,
                       "completed_at": M.utc_now()})
        M.write_json_once(output / "metrics.json", result)
        M.write_json_once(output / "status.json", {"status": result["status"],
                                                   "metrics_sha256": M.sha256(output / "metrics.json")})
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if result["status"] != "complete":
            raise RuntimeError("one or more validation samples failed; no aggregate NLL reported")
    except Exception as error:
        M.write_json_once(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                                    "error": str(error), "time": M.utc_now()})
        if not (output / "status.json").exists():
            M.write_json_once(output / "status.json", {"status": "failed", "failure_file": "failure.json"})
        raise


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--data-file", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--count", type=int, default=PANEL_COUNT)
    prepare_parser.add_argument("--words-per-sample", type=int, default=WORDS_PER_SAMPLE)
    prepare_parser.add_argument("--seed", type=int, default=SEED)
    prepare_parser.set_defaults(function=prepare)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--data-file", required=True)
    evaluate.add_argument("--panel", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--model-name", required=True, choices=sorted(M.MODEL_SPECS))
    evaluate.add_argument("--format", required=True, choices=("int8", "int4"))
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    evaluate.add_argument("--min-free-gpu-gib", type=float, default=DEFAULT_MIN_FREE_GPU_GIB)
    evaluate.set_defaults(function=run)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "max_tokens", MAX_TOKENS) not in range(2, MAX_TOKENS + 1):
        raise ValueError("max-tokens must be between 2 and 512")
    if getattr(args, "min_free_gpu_gib", DEFAULT_MIN_FREE_GPU_GIB) < 0:
        raise ValueError("min-free-gpu-gib must be non-negative")
    args.function(args)


if __name__ == "__main__":
    main()

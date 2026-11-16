"""Sealed final and cross-task evaluations of a frozen RegionAlign campaign.

Preparation reads metadata only. Sealing requires every registered available
source artifact and development report, including negative and rejected results.
Final input is opened only by ``run --split test --final``, after the immutable
selection and all source identities have been checked and access has been logged.
Transfer changes the scoring task without changing a source layout or its bits.
It does not claim that raw coordinates transfer between different checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import os
import time

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M

SCOPE = "regionalign_campaign_evaluation_v1"
SELF = "regionalign/experiments/campaign_evaluation.py"
TARGET_TASK = {"mmlu": "wikitext", "wikitext": "mmlu",
               "okvqa": "textvqa", "textvqa": "okvqa"}


def driver(plan):
    if plan.get("scope") == "llm_regionalign_dev":
        from regionalign.experiments import llm
        return llm
    if plan.get("scope") == "vqa_regionalign_dev":
        from regionalign.experiments import vqa
        return vqa
    raise ValueError("unsupported source protocol")


def frozen_sources(run, relatives):
    destination = Path(run) / "source"
    destination.mkdir()
    records = []
    for relative in sorted(set(relatives)):
        source = C.PROJECT / relative
        saved = destination / relative.replace("/", "__")
        shutil.copyfile(source, saved)
        digest = M.sha256(source)
        if M.sha256(saved) != digest:
            raise RuntimeError("source changed during snapshot")
        records.append({"path": relative, "sha256": digest,
                        "snapshot": str(saved.relative_to(run))})
    return records


def verify_sources(run, plan):
    for record in plan["sources"]:
        if (M.sha256(C.PROJECT / record["path"]) != record["sha256"]
                or M.sha256(Path(run) / record["snapshot"]) != record["sha256"]):
            raise RuntimeError(f"evaluation source drift: {record['path']}")


def source_plan(run):
    run = C.output_path(run)
    raw = M.read_json(run / "plan.json")
    if raw.get("campaign_policy"):
        from regionalign.experiments.campaign_protocol import configure_from_plan
        configure_from_plan(run)
    module = driver(raw)
    _, plan = module.load_plan(run)
    return run, plan, module


def prepare(source_run, output, mode, clean_mmlu_run=None, clean_benchmark_run=None):
    if mode not in ("final", "transfer"):
        raise ValueError("unknown evaluation mode")
    source_run, source, module = source_plan(source_run)
    output = C.output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    if mode == "final":
        if not clean_mmlu_run or not clean_benchmark_run:
            raise ValueError("final preparation needs both existing clean plan paths")
        clean_mmlu_run = C.output_path(clean_mmlu_run)
        clean_benchmark_run = C.output_path(clean_benchmark_run)
        mmlu = M.read_json(clean_mmlu_run / "plan.json")
        benchmarks = M.read_json(clean_benchmark_run / "plan.json")
        if (mmlu.get("scope") != "clean_mmlu_full_parameter_quantization"
                or "dataset" not in mmlu):
            raise ValueError("not an audited MMLU clean plan")
        if benchmarks.get("scope") != "clean_benchmarks_v1":
            raise ValueError("not an audited clean benchmark plan")
        inputs = {"mmlu": mmlu["dataset"], **benchmarks["datasets"],
                  "clean_plan_identities": [
                      {"path": str(clean_mmlu_run / "plan.json"),
                       "sha256": M.sha256(clean_mmlu_run / "plan.json")},
                      {"path": str(clean_benchmark_run / "plan.json"),
                       "sha256": M.sha256(clean_benchmark_run / "plan.json")} ]}
    relatives = [record["path"] for record in source["sources"]]
    relatives += [SELF, "regionalign/experiments/clean_benchmarks.py",
                  "regionalign/runtimes/text.py", "regionalign/runtimes/vqa.py",
                  "regionalign/harness/evidence.py",
                  "regionalign/runtimes/vqa_score.py"]
    conditions = []
    by_key = {(row["model"], row["format"], row["task"]): row for row in source["conditions"]}
    for condition in source["conditions"]:
        target = (condition if mode == "final" else
                  by_key[(condition["model"], condition["format"], TARGET_TASK[condition["task"]])])
        available = (condition.get("availability", "available") == "available"
                     and target.get("availability", "available") == "available")
        conditions.append({"source_condition": condition["id"],
                           "target_condition": target["id"],
                           "model": condition["model"], "format": condition["format"],
                           "source_task": condition["task"], "target_task": target["task"],
                           "availability": "available" if available else "unavailable",
                           "items": [item["name"] for item in module.planned_items(condition, source)]})
    plan = {"schema_version": 1, "scope": SCOPE, "mode": mode,
            "created_at": M.utc_now(), "source_run": str(source_run),
            "source_plan_sha256": source["sha256"], "conditions": conditions,
            "inputs": inputs, "sources": frozen_sources(output, relatives),
            "prior_clean_final_exposure": mode == "final",
            "selection_policy": "all_registered_available_items_without_ranking",
            "target_feedback_to_source_selection": False,
            "cross_checkpoint_coordinate_transfer": False,
            "evaluation_k": list(source["evaluation_k"])}
    M.write_json_once(output / "plan.json", plan)
    return output


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if (plan.get("scope") != SCOPE or plan.get("schema_version") != 1
            or plan.get("mode") not in ("final", "transfer")):
        raise ValueError("wrong evaluation protocol")
    verify_sources(run, plan)
    parent = Path(plan["source_run"])
    if M.sha256(parent / "plan.json") != plan["source_plan_sha256"]:
        raise RuntimeError("source plan changed")
    _, source, module = source_plan(parent)
    return run, plan, source, module


def seal(run):
    run, plan, source, module = load_plan(run)
    records = []
    for condition in plan["conditions"]:
        if condition["availability"] != "available":
            continue
        for name in condition["items"]:
            artifact = Path(plan["source_run"]) / "artifacts" / condition["source_condition"] / name
            development = Path(plan["source_run"]) / "eval" / condition["source_condition"] / name
            if not artifact.is_file() or not development.is_file():
                raise RuntimeError(f"source development is incomplete: {condition['source_condition']}/{name}")
            saved = M.read_json(artifact)
            report = M.read_json(development)
            if (saved.get("plan_sha256") != source["sha256"]
                    or report.get("plan_sha256") != source["sha256"]
                    or report.get("artifact_file_sha256") != M.sha256(artifact)):
                raise RuntimeError("source artifact/development identity mismatch")
            records.append({"condition": condition["source_condition"], "name": name,
                            "artifact": str(artifact), "artifact_sha256": M.sha256(artifact),
                            "development": str(development),
                            "development_sha256": M.sha256(development)})
    selection = {"scope": SCOPE, "mode": plan["mode"],
                 "plan_sha256": M.sha256(run / "plan.json"),
                 "policy": plan["selection_policy"], "records": records,
                 "sealed_at": M.utc_now()}
    M.write_json_once(run / "selection.json", selection)
    return selection


def verify_selection(run, plan):
    selection = M.read_json(run / "selection.json")
    if selection.get("plan_sha256") != M.sha256(run / "plan.json"):
        raise ValueError("selection does not bind this evaluation plan")
    expected = {(c["source_condition"], name) for c in plan["conditions"]
                if c["availability"] == "available" for name in c["items"]}
    actual = {(r["condition"], r["name"]) for r in selection["records"]}
    if actual != expected or len(actual) != len(selection["records"]):
        raise ValueError("selection omits or duplicates registered items")
    for record in selection["records"]:
        if (M.sha256(record["artifact"]) != record["artifact_sha256"]
                or M.sha256(record["development"]) != record["development_sha256"]):
            raise RuntimeError("sealed source evidence changed")
    return selection


def final_access(run, plan, condition):
    record = {"time": M.utc_now(), "scope": SCOPE, "purpose": "fault_final_evaluation",
              "split": "test", "final": True, "run": str(run),
              "plan_sha256": M.sha256(run / "plan.json"),
              "selection_sha256": M.sha256(run / "selection.json"),
              "condition": condition, "pid": os.getpid(), "host": socket.gethostname(),
              "prior_clean_final_exposure": True}
    M.append_jsonl(run / "test_access.jsonl", record)
    M.append_jsonl(C.RESULTS_ROOT / "test_access.jsonl", record)


def final_text_rows(tokenizer, task, inputs):
    from regionalign.experiments import llm as L, clean_benchmarks as B
    if task == "mmlu":
        spec = inputs["mmlu"]
        files = {r["split"]: Path(spec["hf_home"]) / r["path"] for r in spec["files"]}
        for record in spec["files"]:
            if record["split"] in ("dev", "test"):
                if M.sha256(Path(spec["hf_home"]) / record["path"]) != record["sha256"]:
                    raise RuntimeError("final MMLU identity changed")
        demonstrations = M.dev_by_subject(M.read_arrow(files["dev"]))
        choices, choice_tokens = M.choice_token_ids(tokenizer)
        rows = []
        for index, raw in enumerate(M.read_arrow(files["test"])):
            ids, shots = M.prompt_ids(tokenizer, raw["subject"], demonstrations[raw["subject"]],
                                      raw, L.MMLU_MAX_TOKENS, M.DEFAULT_SHOTS)
            rows.append({"sid": f"mmlu-final-{index}", "subject": raw["subject"],
                         "prompt_ids": ids, "answer_ids": [choices[raw["answer"]]],
                         "choice_token_ids": choices, "choice_tokens": choice_tokens,
                         "answer": raw["answer"], "shots": shots,
                         "target_tokens": 1, "objective_weight": 1.0})
        if len(rows) != 14042:
            raise ValueError("final MMLU must cover all 14,042 items")
        return rows
    records = inputs["wikitext"]["files"]
    B.verify_dataset_files(records)
    text = "\n\n".join(row["text"] for row in B.rows_from_arrow(records))
    ids = tokenizer.encode(text, add_special_tokens=False)
    rows = []
    for index, (start, stop, first) in enumerate(B.windows(len(ids), 512, 256)):
        rows.append({"sid": f"wikitext-final-{index}", "prompt_ids": ids[start:first],
                     "answer_ids": ids[first:stop], "gate_prompt_ids": ids[start:stop - 1],
                     "target_tokens": stop - first, "objective_weight": 1.0})
    if sum(r["target_tokens"] for r in rows) != len(ids) - 1:
        raise RuntimeError("final WikiText target coverage is not exact")
    return L._normalise_objective_weights(rows)


def final_vqa_rows(task, inputs):
    from regionalign.experiments import clean_benchmarks as B
    spec = inputs[task]
    B.verify_dataset_files(spec["files"])
    rows = []
    for index, raw in enumerate(B.rows_from_arrow(spec["files"])):
        rows.append({"sid": f"{task}-final-{raw['question_id']}",
                     "question_id": raw["question_id"], "image_id": str(raw["image_id"]),
                     "question": raw["question"], "answers": list(raw["answers"]),
                     "reference": B.reference_answer(raw["answers"], task),
                     "image": raw["image"]})
    if len(rows) != spec["expected_count"] or len({r["sid"] for r in rows}) != len(rows):
        raise ValueError("final VQA coverage is incomplete or duplicated")
    return rows


@contextmanager
def final_vqa_adapter():
    """Allow empty answer-NLL references without discarding generation examples."""
    from regionalign.runtimes import vqa as V
    from regionalign.experiments import vqa as D
    old_prepare, old_metrics = V.HFVQARegionRuntime._prepare_rows, D._task_metrics

    def prepare_rows(self, rows):
        output = [dict(row) for row in rows]
        for row in output:
            row["target_tokens"] = len(self.processor.tokenizer.encode(
                row["reference"], add_special_tokens=False))
            row["objective_weight"] = 1.0
        return output

    def metrics(runtime, rows):
        scored = [row for row in rows if row["target_tokens"]]
        losses = runtime.nll(scored, bs=1)
        finite = all(math.isfinite(value) for value in losses)
        total = sum(row["target_tokens"] for row in scored)
        nll = (sum(value * row["target_tokens"] for row, value in zip(scored, losses)) / total
               if total and finite else None)
        generated = runtime.accuracy(rows)
        records = {row["sid"]: dict(row) for row in generated["records"]}
        for row in rows:
            records[row["sid"]].update(target_tokens=row["target_tokens"],
                                         nll_available=bool(row["target_tokens"]))
        for row, value in zip(scored, losses):
            records[row["sid"]].update(nll=value if math.isfinite(value) else None)
        complete = finite and nll is not None and generated["nonfinite_rate"] == 0
        return {"status": "complete" if complete else "nonfinite", "rows": len(rows),
                "target_tokens": total, "nll": nll, "score": generated["acc"],
                "nonfinite_rate": generated["nonfinite_rate"],
                "eos_rate": generated["eos_rate"], "length_rate": generated["length_rate"],
                "nll_unavailable_samples": len(rows) - len(scored),
                "records": list(records.values())}
    V.HFVQARegionRuntime._prepare_rows, D._task_metrics = prepare_rows, metrics
    try:
        yield
    finally:
        V.HFVQARegionRuntime._prepare_rows, D._task_metrics = old_prepare, old_metrics


def run_condition(run, condition_id, shard="0/1", device="cuda:0", split=None, final=False):
    # Check the explicit final gate before any final input or access log is opened.
    raw = M.read_json(C.output_path(run) / "plan.json")
    if (raw.get("scope") != SCOPE or raw.get("schema_version") != 1
            or raw.get("mode") not in ("final", "transfer")):
        raise ValueError("unsupported evaluation protocol or mode")
    if raw.get("mode") == "final" and (split != "test" or final is not True):
        raise ValueError("final evaluation requires --split test --final")
    if raw.get("mode") == "transfer" and (final or split not in (None, "dev")):
        raise ValueError("cross-task development transfer only accepts --split dev")
    run, plan, source, module = load_plan(run)
    selection = verify_selection(run, plan)
    match = [c for c in plan["conditions"] if c["source_condition"] == condition_id]
    if len(match) != 1 or match[0]["availability"] != "available":
        raise ValueError("condition is absent or unavailable")
    condition = match[0]
    source_condition = module._condition(source, condition_id)
    target_condition = module._condition(source, condition["target_condition"])
    items = module.shard_items(module.planned_items(source_condition, source), shard)
    if plan["mode"] == "final":
        final_access(run, plan, condition_id)
    from regionalign.harness import core as P, region as R, evidence as E
    from regionalign.runtimes.text import FormalTextRuntime
    from contextlib import nullcontext
    is_vqa = source["scope"] == "vqa_regionalign_dev"
    adapter = final_vqa_adapter() if is_vqa and plan["mode"] == "final" else nullcontext()
    with adapter:
        if not is_vqa:
            minimum = source["execution_protocol"]["min_free_gpu_gib"]
            if plan["mode"] == "transfer":
                runtime, rows, audit, timings = module._load_runtime(
                    Path(plan["source_run"]), source, target_condition, "dev", device, minimum)
            else:
                checkpoint = module._verify_checkpoint(source, source_condition)
                model, tokenizer, audit, timings = M.load_model(
                    str(checkpoint), source_condition["format"], device,
                    min_free_gpu_bytes=int(minimum * 1024 ** 3))
                rows = final_text_rows(tokenizer, source_condition["task"], plan["inputs"])
                runtime = FormalTextRuntime(model, tokenizer, {"dev": rows}, device)
                runtime.task = source_condition["task"]
            metric = lambda: module._task_metrics(runtime, rows, target_condition["task"], 1)
        else:
            if plan["mode"] == "transfer":
                runtime, rows = module._load_runtime(
                    Path(plan["source_run"]), source, target_condition, "dev", device)
            else:
                from regionalign.runtimes.vqa import HFVQARegionRuntime
                checkpoint = module._verify_checkpoint(source, source_condition)
                rows = final_vqa_rows(source_condition["task"], plan["inputs"])
                runtime = HFVQARegionRuntime.load(source_condition["model"], str(checkpoint),
                    source_condition["format"], {"dev": rows}, source_condition["task"], device)
                rows = runtime.rows["dev"]
            metric = lambda: module._task_metrics(runtime, rows)
        clean = metric()
        clean["_logits"] = runtime.prefill_logits(rows[:P.LOGIT_ROWS])
        fingerprint = R.model_checksum(runtime.lm)
        registered = {(r["condition"], r["name"]): r for r in selection["records"]}
        try:
            for item in items:
                output = run / "eval" / condition_id / item["name"]
                if output.exists():
                    raise FileExistsError(f"evaluation already exists: {output}")
                with module._item_lock(run, "evaluate", condition_id, item["name"]):
                    verify_sources(run, plan)
                    record = registered[(condition_id, item["name"])]
                    serialized = Path(record["artifact"]).read_bytes()
                    if (hashlib.sha256(serialized).hexdigest() != record["artifact_sha256"]
                            or M.sha256(record["development"]) != record["development_sha256"]):
                        raise RuntimeError("sealed source evidence changed before evaluation")
                    artifact = json.loads(serialized)
                    # Validate the unmodified source before constructing an explicit scoring view.
                    source_reasons = module._artifact_reasons(
                        runtime, artifact, source_condition, item, source)
                    view_condition, view = dict(source_condition), dict(artifact)
                    view_condition["task"] = condition["target_task"]
                    view["task"] = condition["target_task"]
                    if source_reasons:
                        view["error"] = "; ".join(source_reasons)
                    if is_vqa:
                        report = module._evaluate_one(runtime, rows, clean, view_condition, item,
                            view, record["artifact_sha256"], source, print)
                    else:
                        report = module._evaluate_one(runtime, rows, clean, view_condition, item,
                            view, record["artifact_sha256"], source, 1, print)
                    report.update(source_evaluator_scope=report["scope"], scope=SCOPE,
                        mode=plan["mode"], split="test" if final else "dev",
                        final=bool(final), development_only=not final,
                        source_condition=condition_id, target_condition=condition["target_condition"],
                        source_task=condition["source_task"], target_task=condition["target_task"],
                        source_plan_sha256=source["sha256"],
                        evaluation_plan_sha256=M.sha256(run / "plan.json"),
                        selection_sha256=M.sha256(run / "selection.json"),
                        source_artifact_validated_before_scoring_view=True,
                        target_feedback_to_source_selection=False,
                        prior_clean_final_exposure=plan["prior_clean_final_exposure"])
                    report["clean"] = {key: value for key, value in clean.items()
                                       if not key.startswith("_")}
                    report["scoring_view_object_sha256"] = M.object_sha256(view)
                    report["artifact_object_sha256"] = M.object_sha256(artifact)
                    if R.model_checksum(runtime.lm) != fingerprint:
                        raise RuntimeError("model was not restored; report not published")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    M.write_json_once(output, E.sanitize(report))
        finally:
            if hasattr(runtime, "close"):
                runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-run", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--mode", choices=("final", "transfer"), required=True)
    prepare_parser.add_argument("--clean-mmlu-run")
    prepare_parser.add_argument("--clean-benchmark-run")
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--run", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run", required=True)
    run_parser.add_argument("--condition", required=True)
    run_parser.add_argument("--shard", default="0/1")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--split", choices=("dev", "test"))
    run_parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    C.setup_environment()
    if args.command == "prepare":
        print(prepare(args.source_run, args.output, args.mode,
                      args.clean_mmlu_run, args.clean_benchmark_run))
    elif args.command == "seal":
        print(json.dumps({"sealed_items": len(seal(args.run)["records"])}))
    else:
        run_condition(args.run, args.condition, args.shard, args.device, args.split, args.final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

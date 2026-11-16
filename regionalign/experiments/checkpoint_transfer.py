"""Sealed, development-only Llama-2 base to chat checkpoint transfer.

This control applies every registered source layout and bit list verbatim to a
different checkpoint with exactly the same architecture and token vocabulary.
There is no coordinate mapping, optimization, gradient, chat templating, or
target-dependent source selection. All four INT8/INT4 x MMLU/WikiText source
conditions and their 108 artifact/dev pairs, including errors and negative
results, must be sealed before execution. Only the source frozen dev token IDs
are used. Target clean metrics, layout gates and bit legality are measured anew.

Source checkpoint manifests and saved artifact/dev hashes identify the source;
they are not described as a measured source runtime checksum. Only the target
runtime is loaded, with an exact bounded checksum before and after each item.
The original plan's snapshots and this module's executing sources are separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import campaign_evaluation as E
from regionalign.experiments import campaign_memory as CM
from regionalign.experiments import campaign_protocol as B
from regionalign.experiments import clean_mmlu as M
from regionalign.experiments import llm as L
from regionalign.harness import core as P, evidence as J, region as R
from regionalign.runtimes.text import FormalTextRuntime

SCOPE = "regionalign_checkpoint_transfer_dev_v1"
SELF = "regionalign/experiments/checkpoint_transfer.py"
SOURCE_MODEL = "llama2-7b"
TARGET_MODEL = "llama2-7b-chat"
SELECTION_POLICY = "all_four_source_conditions_all_108_registered_items_without_ranking"
COORDINATES = "identity_same_architecture_original_layout_and_flips_verbatim"
TOKENIZERS = ("tokenizer.model", "tokenizer.json")
CONFIG_IDENTITY_KEYS = ("_name_or_path", "transformers_version")


def _digest_body(manifest):
    return M.object_sha256({key: value for key, value in manifest.items()
                            if key != "manifest_sha256"})


def _parent(run):
    run, plan, digest = B._read_registered_plan(C.output_path(run))
    if plan.get("scope") != L.SCOPE:
        raise ValueError("checkpoint transfer requires a language development campaign")
    policy = B._validate_policy(run, plan)
    if (policy["variant"] != "baseline" or plan["search_budget"] != R.BUDGET
            or plan.get("execution_verification") != CM.POLICY
            or policy.get("verification_sha256") != M.object_sha256(CM.POLICY)
            or plan.get("arms") != list(L.ARMS)
            or plan.get("repeat_seeds") != list(L.REPEAT_SEEDS)
            or plan.get("work_items_per_condition") != 108
            or plan.get("evaluation_k") != list(L.EVAL_K)
            or plan.get("search_k") != list(R.K_SCHEDULE)
            or plan.get("clean_gate_nll") != R.GATE_NLL
            or plan.get("mmlu_score_gate") != L.MMLU_SCORE_GATE):
        raise ValueError("the frozen baseline and exact checksum protocol are required")
    L._assert_frozen(run, plan)
    plan["sha256"] = digest
    manifest = plan["models"][SOURCE_MODEL]
    if manifest.get("manifest_sha256") != _digest_body(manifest):
        raise ValueError("source checkpoint manifest embedded digest differs")
    return run, plan


def target_manifest(checkpoint, source_manifest):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint == Path(source_manifest["path"]).resolve():
        raise ValueError("target must be a different checkpoint")
    source_records = {record["path"]: record for record in source_manifest["files"]}
    source_config_bytes = (Path(source_manifest["path"]) / "config.json").read_bytes()
    if ("config.json" not in source_records or hashlib.sha256(source_config_bytes).hexdigest()
            != source_records["config.json"]["sha256"]):
        raise ValueError("source configuration differs from its frozen manifest")
    source_config = json.loads(source_config_bytes)
    config_bytes = (checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    semantic = lambda value: {k: v for k, v in value.items() if k not in CONFIG_IDENTITY_KEYS}
    if semantic(config) != semantic(source_config):
        raise ValueError("source and target architecture/configuration differ")
    files = [M.file_record(path, checkpoint) for path in M.checkpoint_files(checkpoint)]
    target_records = {record["path"]: record for record in files}
    if hashlib.sha256(config_bytes).hexdigest() != target_records["config.json"]["sha256"]:
        raise RuntimeError("target configuration changed while freezing")
    for name in TOKENIZERS:
        if (name not in source_records or name not in target_records
                or source_records[name]["sha256"] != target_records[name]["sha256"]
                or M.sha256(Path(source_manifest["path"]) / name) != source_records[name]["sha256"]):
            raise ValueError(f"source/target {name} bytes must be identical")
    source_weights = sorted(record["sha256"] for name, record in source_records.items()
                            if name.endswith(".safetensors"))
    target_weights = sorted(record["sha256"] for name, record in target_records.items()
                            if name.endswith(".safetensors"))
    if not source_weights or source_weights == target_weights:
        raise ValueError("target checkpoint must have different weight bytes")
    manifest = {"path": str(checkpoint), "model_key": TARGET_MODEL,
                "config": config, "files": files, "tokenizer_identity_files": list(TOKENIZERS),
                "tokenizer_identity": "byte_identical_to_source_no_retokenization_or_chat_template",
                "source_checkpoint_manifest_sha256": source_manifest["manifest_sha256"]}
    manifest["manifest_sha256"] = M.object_sha256(manifest)
    return manifest


def _conditions(source, target):
    found = []
    for fmt in L.FORMATS:
        for task in L.TASKS:
            matches = [c for c in source["conditions"]
                       if (c["model"], c["format"], c["task"]) == (SOURCE_MODEL, fmt, task)]
            if len(matches) != 1 or matches[0].get("availability", "available") != "available":
                raise ValueError(f"missing/duplicate/unavailable source {fmt}/{task}")
            condition = matches[0]
            if (condition.get("checkpoint_manifest_sha256") != source["models"][SOURCE_MODEL]["manifest_sha256"]
                    or condition["dims"] != L.dims_for_format(target["config"], fmt)):
                raise ValueError("source manifest binding or exact target dimensions differ")
            items = L.planned_items(condition, source)
            if len(items) != 108 or len({i["name"] for i in items}) != 108:
                raise ValueError("each source condition must register exactly 108 distinct items")
            found.append({"source_condition": condition["id"], "source_model": SOURCE_MODEL,
                "target_model": TARGET_MODEL, "format": fmt, "task": task,
                "source_checkpoint_manifest_sha256": condition["checkpoint_manifest_sha256"],
                "target_checkpoint_manifest_sha256": target["manifest_sha256"],
                "dims": copy.deepcopy(condition["dims"]), "regions_sha256": M.object_sha256(condition["regions"]),
                "panels": copy.deepcopy(condition["panels"]), "items": [i["name"] for i in items]})
    return found


def _copy_once(source, target):
    data = Path(source).read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def prepare(source_run, target_checkpoint, output):
    source_run, source = _parent(source_run)
    target = target_manifest(target_checkpoint, source["models"][SOURCE_MODEL])
    conditions = _conditions(source, target)
    output = C.output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    if _copy_once(source_run / "plan.json", output / "parent" / "plan.json") != source["sha256"]:
        raise RuntimeError("source plan changed while copying")
    historical = []
    for index, record in enumerate(source["sources"]):
        saved = output / "parent" / "source" / f"{index:03d}__{Path(record['snapshot']).name}"
        if _copy_once(source_run / record["snapshot"], saved) != record["sha256"]:
            raise RuntimeError("historical source changed while copying")
        historical.append({"original_path": record["path"], "original_snapshot": record["snapshot"],
                           "snapshot": str(saved.relative_to(output)), "sha256": record["sha256"]})
    relatives = [r["path"] for r in source["sources"]]
    relatives += [SELF, E.SELF, CM.SELF, CM.BOUNDED, "regionalign/harness/evidence.py"]
    plan = {"schema_version": 1, "scope": SCOPE, "created_at": M.utc_now(),
            "development_only": True, "splits": ["dev"], "source_run": str(source_run),
            "source_plan_sha256": source["sha256"], "source_model": SOURCE_MODEL, "target_model": TARGET_MODEL,
            "target_checkpoint": target, "conditions": conditions, "historical_sources": historical,
            "sources": E.frozen_sources(output, relatives), "selection_policy": SELECTION_POLICY,
            "coordinate_policy": COORDINATES, "target_feedback_to_source_selection": False,
            "evaluation_k": list(L.EVAL_K), "batch_size": 1}
    E.verify_sources(output, plan)
    if _parent(source_run)[1]["sha256"] != source["sha256"]:
        raise RuntimeError("source plan changed during preparation")
    M.write_json_once(output / "plan.json", plan)
    return output


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if (plan.get("schema_version") != 1 or plan.get("scope") != SCOPE
            or plan.get("development_only") is not True or plan.get("splits") != ["dev"]
            or plan.get("selection_policy") != SELECTION_POLICY
            or plan.get("coordinate_policy") != COORDINATES
            or plan.get("target_feedback_to_source_selection") is not False
            or plan.get("batch_size") != 1 or plan.get("evaluation_k") != list(L.EVAL_K)):
        raise ValueError("not the sealed development-only same-architecture checkpoint control")
    E.verify_sources(run, plan)
    source_run, source = _parent(plan["source_run"])
    if (source["sha256"] != plan["source_plan_sha256"]
            or M.sha256(run / "parent" / "plan.json") != source["sha256"]
            or _conditions(source, plan["target_checkpoint"]) != plan["conditions"]
            or plan["target_checkpoint"].get("manifest_sha256") != _digest_body(plan["target_checkpoint"])):
        raise RuntimeError("source, target or transfer registration changed")
    if [(r["original_path"], r["original_snapshot"], r["sha256"]) for r in plan["historical_sources"]] != [
            (r["path"], r["snapshot"], r["sha256"]) for r in source["sources"]]:
        raise RuntimeError("historical source identities changed")
    if any(M.sha256(run / r["snapshot"]) != r["sha256"] for r in plan["historical_sources"]):
        raise RuntimeError("historical source snapshots changed")
    return run, plan, source


def _evidence(source_run, source, condition, item):
    artifact = source_run / "artifacts" / condition["id"] / item["name"]
    development = source_run / "eval" / condition["id"] / item["name"]
    if not artifact.is_file() or not development.is_file():
        raise RuntimeError(f"source development incomplete: {condition['id']}/{item['name']}")
    data, dev_data = artifact.read_bytes(), development.read_bytes()
    saved, report = json.loads(data), json.loads(dev_data)
    digest = hashlib.sha256(data).hexdigest()
    expected = {"schema_version": L.SCHEMA_VERSION, "scope": L.SCOPE,
        "plan_sha256": source["sha256"], "condition": condition["id"], "region": item["region"],
        "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
        "campaign_scope": B.CAMPAIGN_SCOPE, "campaign_variant": "baseline",
        "campaign_policy_sha256": M.object_sha256(source["campaign_policy"])}
    if (any(saved.get(k) != v or report.get(k) != v for k, v in expected.items())
            or report.get("artifact_file_sha256") != digest or not isinstance(report.get("status"), str)):
        raise RuntimeError("source artifact/development identity mismatch")
    return {"condition": condition["id"], "name": item["name"], "artifact": str(artifact),
            "artifact_sha256": digest, "development": str(development),
            "development_sha256": hashlib.sha256(dev_data).hexdigest(),
            "development_status": report["status"], "source_error": saved.get("error")}


def _all_evidence(plan, source):
    records = []
    for registration in plan["conditions"]:
        condition = L._condition(source, registration["source_condition"])
        for item in L.planned_items(condition, source):
            records.append(_evidence(Path(plan["source_run"]), source, condition, item))
    return records


def seal(run):
    run, plan, source = load_plan(run)
    selection = {"schema_version": 1, "scope": SCOPE, "split": "dev", "sealed_at": M.utc_now(),
        "plan_sha256": M.sha256(run / "plan.json"), "source_plan_sha256": source["sha256"],
        "target_checkpoint_manifest_sha256": plan["target_checkpoint"]["manifest_sha256"],
        "policy": SELECTION_POLICY, "target_feedback_to_source_selection": False,
        "records": _all_evidence(plan, source)}
    M.write_json_once(run / "selection.json", selection)
    return selection


def verify_selection(run, plan, source):
    selection = M.read_json(run / "selection.json")
    if (selection.get("scope") != SCOPE or selection.get("split") != "dev"
            or selection.get("policy") != SELECTION_POLICY
            or selection.get("target_feedback_to_source_selection") is not False
            or selection.get("plan_sha256") != M.sha256(run / "plan.json")
            or selection.get("source_plan_sha256") != source["sha256"]
            or selection.get("target_checkpoint_manifest_sha256") != plan["target_checkpoint"]["manifest_sha256"]
            or selection.get("records") != _all_evidence(plan, source)):
        raise RuntimeError("selection or sealed source evidence changed")
    return selection


def _assert_bound(run, plan, transfer_sha, selection_sha):
    if any(M.sha256(path) != digest for path, digest in (
            (run / "plan.json", transfer_sha), (run / "selection.json", selection_sha),
            (Path(plan["source_run"]) / "plan.json", plan["source_plan_sha256"]))):
        raise RuntimeError("source, transfer or selection changed during execution")


def _load_target(plan, source, condition, device):
    frozen = plan["target_checkpoint"]
    if target_manifest(frozen["path"], source["models"][SOURCE_MODEL]) != frozen:
        raise RuntimeError("target checkpoint changed after preparation")
    rows = L._panel_rows(Path(plan["source_run"]), condition, "dev")
    model, tokenizer, audit, timings = M.load_model(frozen["path"], condition["format"], device,
        min_free_gpu_bytes=int(source["execution_protocol"]["min_free_gpu_gib"] * 1024 ** 3))
    if target_manifest(frozen["path"], source["models"][SOURCE_MODEL]) != frozen:
        raise RuntimeError("target checkpoint changed while loading")
    if P.dims(model) != condition["dims"]:
        raise RuntimeError("loaded target dimensions differ from the frozen source")
    runtime = FormalTextRuntime(model, tokenizer, {"dev": rows}, device)
    runtime.task = condition["task"]
    return runtime, rows, audit, timings


def run_condition(run, condition_id, shard="0/36", device="cuda:0", split="dev"):
    if split != "dev":
        raise ValueError("checkpoint transfer only accepts --split dev")
    run, plan, source = load_plan(run)
    selection = verify_selection(run, plan, source)
    selection_bytes = (run / "selection.json").read_bytes()
    if json.loads(selection_bytes) != selection:
        raise RuntimeError("selection changed while being read")
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    transfer_sha = selection["plan_sha256"]
    _assert_bound(run, plan, transfer_sha, selection_sha)
    if condition_id not in {c["source_condition"] for c in plan["conditions"]}:
        raise ValueError("unknown source condition")
    condition = L._condition(source, condition_id)
    if shard.split("/")[-1] != "36":
        raise ValueError("checkpoint transfer requires INDEX/36 matched-group shards")
    items = L.shard_items(L.planned_items(condition, source), shard)
    if len(items) != 3:
        raise ValueError("each shard must contain exactly three matched arms")
    configured = B.configure_from_plan(plan["source_run"])
    if configured["sha256"] != source["sha256"]:
        raise RuntimeError("configured source identity differs")
    registered = {(r["condition"], r["name"]): r for r in selection["records"]}
    with CM.installed(plan["source_run"]):
        runtime, rows, audit, timings = _load_target(plan, source, condition, device)
        try:
            if runtime.task != condition["task"] or P.dims(runtime.lm) != condition["dims"]:
                raise RuntimeError("target runtime task or dimensions differ")
            fingerprint = R.model_checksum(runtime.lm)
            clean = L._task_metrics(runtime, rows, condition["task"], 1)
            clean["_logits"] = runtime.prefill_logits(rows[:P.LOGIT_ROWS])
            clean.update(logits_sha256=L._logits_digest(clean["_logits"]), quantization=audit, load_timings=timings)
            for item in items:
                output = run / "eval" / condition_id / item["name"]
                with L._item_lock(run, "evaluate", condition_id, item["name"]):
                    if output.exists():
                        raise FileExistsError(f"refusing to overwrite checkpoint transfer report: {output}")
                    _assert_bound(run, plan, transfer_sha, selection_sha)
                    E.verify_sources(run, plan)
                    record = registered[(condition_id, item["name"])]
                    if _evidence(Path(plan["source_run"]), source, condition, item) != record:
                        raise RuntimeError("sealed source evidence changed before evaluation")
                    artifact_bytes = Path(record["artifact"]).read_bytes()
                    if hashlib.sha256(artifact_bytes).hexdigest() != record["artifact_sha256"]:
                        raise RuntimeError("source artifact changed while being read")
                    artifact = json.loads(artifact_bytes)
                    source_dev_bytes = Path(record["development"]).read_bytes()
                    if hashlib.sha256(source_dev_bytes).hexdigest() != record["development_sha256"]:
                        raise RuntimeError("source development changed while being read")
                    source_dev = json.loads(source_dev_bytes)
                    original_object = M.object_sha256(artifact)
                    # Validate under the ORIGINAL source task/plan. Nothing in the
                    # artifact is rewritten to make it acceptable to the target.
                    reasons = L._artifact_reasons(runtime, artifact, condition, item, source)
                    report = L._evaluate_one(runtime, rows, clean, condition, item, artifact,
                                              record["artifact_sha256"], source, 1, print)
                    after = R.model_checksum(runtime.lm)
                    if after != fingerprint:
                        raise RuntimeError("target model was not restored; report not published")
                    if M.object_sha256(artifact) != original_object:
                        raise RuntimeError("evaluator changed the source artifact in memory")
                    if _evidence(Path(plan["source_run"]), source, condition, item) != record:
                        raise RuntimeError("source evidence changed during evaluation")
                    _assert_bound(run, plan, transfer_sha, selection_sha)
                    report.update(source_evaluator_scope=report["scope"], scope=SCOPE,
                        split="dev", development_only=True, source_model=SOURCE_MODEL, target_model=TARGET_MODEL,
                        source_run=plan["source_run"], source_condition=condition_id,
                        source_plan_sha256=source["sha256"], transfer_plan_sha256=transfer_sha,
                        selection_sha256=selection_sha, coordinate_policy=COORDINATES,
                        source_checkpoint_manifest_sha256=condition["checkpoint_manifest_sha256"],
                        target_checkpoint_manifest_sha256=plan["target_checkpoint"]["manifest_sha256"],
                        source_development_sha256=record["development_sha256"],
                        source_development_status=record["development_status"],
                        source_development_metrics={k: source_dev[k] for k in ("clean", "rearranged", "results", "summary") if k in source_dev},
                        source_artifact_reasons=reasons, source_artifact_validated_under_original_plan=True,
                        source_validation_scope="original_plan_metadata_and_archived_dev_no_source_runtime_reload",
                        source_runtime_loaded=False, target_bit_legality_rechecked=True,
                        artifact_object_sha256=original_object, target_feedback_to_source_selection=False,
                        target_runtime_checksum_before=fingerprint, target_runtime_checksum_after=after,
                        clean={k: v for k, v in clean.items() if not k.startswith("_")})
                    report = J.sanitize(report)
                    json.dumps(report, allow_nan=False)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    M.write_json_once(output, report)
        finally:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source-run", required=True)
    p.add_argument("--target-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("seal"); p.add_argument("--run", required=True)
    p = commands.add_parser("run"); p.add_argument("--run", required=True)
    p.add_argument("--condition", required=True); p.add_argument("--shard", default="0/36")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--split", choices=("dev",), required=True)
    args = parser.parse_args(argv); C.setup_environment()
    if args.command == "prepare":
        print(prepare(args.source_run, args.target_checkpoint, args.output))
    elif args.command == "seal":
        print(json.dumps({"sealed_items": len(seal(args.run)["records"])}))
    else:
        run_condition(args.run, args.condition, args.shard, args.device, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

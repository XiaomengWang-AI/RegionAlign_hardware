"""Sealed VQA task transfer between two independently frozen dev campaigns.

Both directions evaluate every registered source item, including rejected or
negative results; target scores never rank or select source attacks. Preparation
hashes train-derived panel files without opening their rows. Sealing requires all
432 source artifact/dev pairs. Execution configures only the source campaign,
verifies the target under that identical resource policy, and opens only target
dev rows. Source identities are checked before a task-only scoring view is made.
The original layout and flips remain unchanged. There is no test/final interface.

Historical parent source snapshots and the actual executing source snapshots are
recorded separately. A source snapshot cannot stand in for different live code.
Each process must use one source campaign; no configuration reset is performed.
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
from regionalign.experiments import vqa as V
from regionalign.harness import core as P, evidence as J, region as R

SCOPE = "regionalign_vqa_task_transfer_dev_v1"
SELF = "regionalign/experiments/task_transfer.py"
OPPOSITE = {"okvqa": "textvqa", "textvqa": "okvqa"}
SELECTION_POLICY = "all_four_source_conditions_all_108_registered_items_without_ranking"


def _parent(run):
    run, plan, digest = B._read_registered_plan(C.output_path(run))
    if plan["scope"] != V.SCOPE:
        raise ValueError("VQA task transfer requires VQA development campaigns")
    policy = B._validate_policy(run, plan)
    if (policy["variant"] != "baseline" or plan["search_budget"] != 300
            or plan.get("execution_verification") != CM.POLICY
            or policy.get("verification_sha256") != M.object_sha256(CM.POLICY)):
        raise ValueError("a baseline with the frozen exact checksum policy is required")
    expected_execution = {"search_batch_size": V.SEARCH_BATCH_SIZE,
                          "grad_batch_size": V.GRAD_BATCH_SIZE,
                          "eval_batch_size": V.EVAL_BATCH_SIZE,
                          "min_free_gpu_gib": 22.0, "min_free_policy": V.MIN_FREE_POLICY,
                          "real_gpu_preflight_required": True}
    if plan.get("execution") != expected_execution:
        raise ValueError("task transfer requires the frozen 22 GiB VQA resource policy")
    # This checks actual live code as well as each original saved snapshot.
    V._assert_frozen(run, plan)
    plan["sha256"] = digest
    return run, plan


def _conditions(source, source_task, target, target_task):
    if source_task not in OPPOSITE or OPPOSITE[source_task] != target_task:
        raise ValueError("only opposite OK-VQA/TextVQA tasks are supported")
    fields = ("execution", "execution_verification", "arms", "repeat_seeds", "search_budget",
              "search_cost", "search_k", "evaluation_k", "clean_gate_nll", "vqa_score_gate",
              "success_relative_score_drop", "work_items_per_condition")
    if any(source.get(key) != target.get(key) for key in fields):
        raise ValueError("source and target scientific/resource protocols differ")
    if (source.get("arms") != list(V.ARMS) or source.get("repeat_seeds") != list(V.REPEAT_SEEDS)
            or source.get("work_items_per_condition") != 108
            or source.get("evaluation_k") != list(V.EVAL_K)
            or source.get("search_k") != list(R.K_SCHEDULE)
            or source.get("search_cost") != R.COST
            or source.get("clean_gate_nll") != R.GATE_NLL
            or source.get("vqa_score_gate") != V.VQA_SCORE_GATE
            or source.get("success_relative_score_drop") != V.VQA_SUCCESS_DROP):
        raise ValueError("not the formal four-condition three-arm VQA baseline")
    found = []
    for model in V.DEV_MODELS:
        for fmt in V.FORMATS:
            sides = []
            for plan, task in ((source, source_task), (target, target_task)):
                rows = [row for row in plan["conditions"] if
                        (row["model"], row["format"], row["task"]) == (model, fmt, task)]
                if len(rows) != 1 or rows[0].get("availability") != "available":
                    raise ValueError(f"missing/duplicate/unavailable {model}/{fmt}/{task}")
                sides.append(rows[0])
            left, right = sides
            manifest = source["models"][model]
            manifest_hash = M.object_sha256({key: value for key, value in manifest.items()
                                            if key != "manifest_sha256"})
            if (manifest != target["models"][model]
                    or manifest.get("manifest_sha256") != manifest_hash
                    or left.get("checkpoint_manifest_sha256") != manifest_hash
                    or right.get("checkpoint_manifest_sha256") != manifest_hash
                    or left["dims"] != right["dims"] or left["regions"] != right["regions"]):
                raise ValueError("checkpoint, quantization dimensions or registered regions differ")
            items = V.planned_items(left, source)
            if len(items) != 108 or len({item["name"] for item in items}) != 108:
                raise ValueError("source condition must register exactly 108 distinct items")
            found.append({"source_condition": left["id"], "target_condition": right["id"],
                "model": model, "format": fmt, "source_task": source_task, "target_task": target_task,
                "checkpoint_manifest_sha256": left["checkpoint_manifest_sha256"],
                "dims_sha256": M.object_sha256(left["dims"]),
                "regions_sha256": M.object_sha256(left["regions"]),
                "source_panels": copy.deepcopy(left["panels"]), "target_panels": copy.deepcopy(right["panels"]),
                "items": [item["name"] for item in items]})
    return found


def _copy_once(source, target):
    serialized = Path(source).read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(serialized)
    return hashlib.sha256(serialized).hexdigest()


def prepare(source_run, source_task, target_run, target_task, output):
    source_run, source = _parent(source_run)
    target_run, target = _parent(target_run)
    conditions = _conditions(source, source_task, target, target_task)
    output = C.output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    parents = []
    for role, run, plan in (("source", source_run, source), ("target", target_run, target)):
        saved = output / "parents" / f"{role}_plan.json"
        if _copy_once(run / "plan.json", saved) != plan["sha256"]:
            raise RuntimeError("parent plan changed during preparation")
        records = []
        for index, record in enumerate(plan["sources"]):
            snapshot = output / "parents" / role / f"{index:03d}__{Path(record['snapshot']).name}"
            if _copy_once(run / record["snapshot"], snapshot) != record["sha256"]:
                raise RuntimeError("parent source changed during preparation")
            records.append({"original_path": record["path"], "original_snapshot": record["snapshot"],
                            "snapshot": str(snapshot.relative_to(output)), "sha256": record["sha256"]})
        parents.append({"role": role, "run": str(run), "plan_sha256": plan["sha256"],
                        "plan_snapshot": str(saved.relative_to(output)), "sources": records})
    relatives = [record["path"] for plan in (source, target) for record in plan["sources"]]
    relatives += [SELF, E.SELF, CM.SELF, CM.BOUNDED, "regionalign/harness/evidence.py"]
    plan = {"schema_version": 1, "scope": SCOPE, "created_at": M.utc_now(),
            "development_only": True, "splits": ["dev"], "source_run": str(source_run),
            "target_run": str(target_run), "source_plan_sha256": source["sha256"],
            "target_plan_sha256": target["sha256"], "source_task": source_task, "target_task": target_task,
            "conditions": conditions, "parents": parents, "sources": E.frozen_sources(output, relatives),
            "selection_policy": SELECTION_POLICY, "target_feedback_to_source_selection": False,
            "cross_checkpoint_coordinate_transfer": False, "evaluation_k": list(source["evaluation_k"])}
    E.verify_sources(output, plan)
    # Recheck identities after copying; preparation never configures a runtime.
    if (_parent(source_run)[1]["sha256"] != source["sha256"]
            or _parent(target_run)[1]["sha256"] != target["sha256"]):
        raise RuntimeError("parent changed during preparation")
    M.write_json_once(output / "plan.json", plan)
    return output


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if (plan.get("schema_version") != 1 or plan.get("scope") != SCOPE
            or plan.get("development_only") is not True or plan.get("splits") != ["dev"]
            or plan.get("selection_policy") != SELECTION_POLICY
            or plan.get("target_feedback_to_source_selection") is not False
            or plan.get("cross_checkpoint_coordinate_transfer") is not False):
        raise ValueError("not a development-only VQA task transfer plan")
    E.verify_sources(run, plan)
    source_run, source = _parent(plan["source_run"])
    target_run, target = _parent(plan["target_run"])
    if (source["sha256"] != plan["source_plan_sha256"] or target["sha256"] != plan["target_plan_sha256"]
            or _conditions(source, plan["source_task"], target, plan["target_task"]) != plan["conditions"]):
        raise RuntimeError("parent plans or registered transfer conditions changed")
    if len(plan["parents"]) != 2 or {p["role"] for p in plan["parents"]} != {"source", "target"}:
        raise ValueError("both historical source provenances are required")
    for parent in plan["parents"]:
        expected = source if parent["role"] == "source" else target
        expected_run = source_run if parent["role"] == "source" else target_run
        if (parent["run"] != str(expected_run) or parent["plan_sha256"] != expected["sha256"]
                or M.sha256(run / parent["plan_snapshot"]) != expected["sha256"]
                or [(r["original_path"], r["original_snapshot"], r["sha256"]) for r in parent["sources"]]
                   != [(r["path"], r["snapshot"], r["sha256"]) for r in expected["sources"]]
                or any(M.sha256(run / r["snapshot"]) != r["sha256"] for r in parent["sources"])):
            raise RuntimeError("historical parent provenance changed")
    return run, plan, source, target


def _evidence(run, source, condition, item):
    artifact = run / "artifacts" / condition["id"] / item["name"]
    development = run / "eval" / condition["id"] / item["name"]
    if not artifact.is_file() or not development.is_file():
        raise RuntimeError(f"source development is incomplete: {condition['id']}/{item['name']}")
    serialized = artifact.read_bytes()
    saved = json.loads(serialized)
    artifact_hash = hashlib.sha256(serialized).hexdigest()
    report_bytes = development.read_bytes()
    report = json.loads(report_bytes)
    expected = {"schema_version": 2, "scope": V.SCOPE, "plan_sha256": source["sha256"],
                "condition": condition["id"], "region": item["region"], "repeat": item["repeat"],
                "seed": item["seed"], "arm": item["arm"], "campaign_scope": B.CAMPAIGN_SCOPE,
                "campaign_variant": "baseline", "campaign_policy_sha256": M.object_sha256(source["campaign_policy"])}
    if (any(saved.get(key) != value or report.get(key) != value for key, value in expected.items())
            or report.get("artifact_file_sha256") != artifact_hash or not isinstance(report.get("status"), str)):
        raise RuntimeError("source artifact/development identity mismatch")
    return {"condition": condition["id"], "name": item["name"], "artifact": str(artifact),
            "artifact_sha256": artifact_hash, "development": str(development),
            "development_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "development_status": report["status"], "source_error": saved.get("error")}


def _all_evidence(plan, source):
    records = []
    for registration in plan["conditions"]:
        condition = V._condition(source, registration["source_condition"])
        for item in V.planned_items(condition, source):
            records.append(_evidence(Path(plan["source_run"]), source, condition, item))
    return records


def seal(run):
    run, plan, source, _ = load_plan(run)
    records = _all_evidence(plan, source)
    selection = {"schema_version": 1, "scope": SCOPE, "split": "dev", "sealed_at": M.utc_now(),
                 "plan_sha256": M.sha256(run / "plan.json"), "source_plan_sha256": source["sha256"],
                 "target_plan_sha256": plan["target_plan_sha256"], "policy": SELECTION_POLICY,
                 "target_feedback_to_source_selection": False, "records": records}
    M.write_json_once(run / "selection.json", selection)
    return selection


def verify_selection(run, plan, source):
    selection = M.read_json(run / "selection.json")
    if (selection.get("schema_version") != 1 or selection.get("scope") != SCOPE
            or selection.get("split") != "dev" or selection.get("policy") != SELECTION_POLICY
            or selection.get("target_feedback_to_source_selection") is not False
            or selection.get("plan_sha256") != M.sha256(run / "plan.json")
            or selection.get("source_plan_sha256") != source["sha256"]
            or selection.get("target_plan_sha256") != plan["target_plan_sha256"]
            or selection.get("records") != _all_evidence(plan, source)):
        raise RuntimeError("selection or sealed source evidence changed")
    return selection


def _assert_bound(run, plan, transfer_sha256, selection_sha256):
    paths = ((run / "plan.json", transfer_sha256), (run / "selection.json", selection_sha256),
             (Path(plan["source_run"]) / "plan.json", plan["source_plan_sha256"]),
             (Path(plan["target_run"]) / "plan.json", plan["target_plan_sha256"]))
    if any(M.sha256(path) != digest for path, digest in paths):
        raise RuntimeError("transfer, selection or parent plan changed during execution")


def run_condition(run, condition_id, shard="0/36", device="cuda:0", split="dev"):
    if split != "dev":
        raise ValueError("task transfer only accepts --split dev")
    run, plan, source, target = load_plan(run)
    selection = verify_selection(run, plan, source)
    selection_bytes = (run / "selection.json").read_bytes()
    if json.loads(selection_bytes) != selection:
        raise RuntimeError("selection changed while being read")
    transfer_sha256 = selection["plan_sha256"]
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    _assert_bound(run, plan, transfer_sha256, selection_sha256)
    found = [row for row in plan["conditions"] if row["source_condition"] == condition_id]
    if len(found) != 1:
        raise ValueError("unknown source condition")
    condition = found[0]
    source_condition = V._condition(source, condition_id)
    target_condition = V._condition(target, condition["target_condition"])
    if shard.split("/")[-1] != "36":
        raise ValueError("task transfer requires INDEX/36 matched-group shards")
    items = V.shard_items(V.planned_items(source_condition, source), shard)
    if len(items) != 3:
        raise ValueError("each shard must contain one matched three-arm group")
    # Never configure the target: the source patch is already the same policy.
    configured = B.configure_from_plan(plan["source_run"])
    if configured["sha256"] != source["sha256"]:
        raise RuntimeError("configured source identity differs")
    _, verified_target = V.load_plan(plan["target_run"])
    if verified_target["sha256"] != target["sha256"]:
        raise RuntimeError("target identity changed during configuration")
    V._require_preflight(Path(plan["source_run"]), source, source_condition)
    V._require_preflight(Path(plan["target_run"]), target, target_condition)
    registered = {(row["condition"], row["name"]): row for row in selection["records"]}
    with CM.installed(plan["source_run"]):
        runtime, rows = V._load_runtime(Path(plan["target_run"]), target, target_condition, "dev", device)
        try:
            _assert_bound(run, plan, transfer_sha256, selection_sha256)
            if runtime.task != plan["target_task"]:
                raise RuntimeError("runtime scoring task differs from frozen target")
            clean = V._task_metrics(runtime, rows)
            clean["_logits"] = runtime.prefill_logits(rows[:P.LOGIT_ROWS])
            fingerprint = R.model_checksum(runtime.lm)
            for item in items:
                output = run / "eval" / condition_id / item["name"]
                with V._item_lock(run, "evaluate", condition_id, item["name"]):
                    _assert_bound(run, plan, transfer_sha256, selection_sha256)
                    E.verify_sources(run, plan)
                    record = registered[(condition_id, item["name"])]
                    if _evidence(Path(plan["source_run"]), source, source_condition, item) != record:
                        raise RuntimeError("sealed source evidence changed before scoring")
                    if output.exists():
                        raise FileExistsError(f"refusing to overwrite transfer report: {output}")
                    serialized = Path(record["artifact"]).read_bytes()
                    if hashlib.sha256(serialized).hexdigest() != record["artifact_sha256"]:
                        raise RuntimeError("source changed while reading scoring artifact")
                    artifact = json.loads(serialized)
                    reasons = V._artifact_reasons(runtime, artifact, source_condition, item, source)
                    view, scoring_condition = copy.deepcopy(artifact), copy.deepcopy(source_condition)
                    view["task"] = scoring_condition["task"] = plan["target_task"]
                    if reasons:
                        view["error"] = "; ".join(str(reason) for reason in reasons)
                    report = V._evaluate_one(runtime, rows, clean, scoring_condition, item, view,
                                             record["artifact_sha256"], source, print)
                    report.update(source_evaluator_scope=report["scope"], scope=SCOPE,
                        split="dev", development_only=True, source_task=plan["source_task"],
                        target_task=plan["target_task"], source_condition=condition_id,
                        target_condition=condition["target_condition"], source_run=plan["source_run"],
                        target_run=plan["target_run"], source_plan_sha256=source["sha256"],
                        target_plan_sha256=target["sha256"], transfer_plan_sha256=transfer_sha256,
                        selection_sha256=selection_sha256,
                        source_artifact_validated_before_scoring_view=True,
                        target_feedback_to_source_selection=False,
                        artifact_object_sha256=M.object_sha256(artifact),
                        scoring_view_object_sha256=M.object_sha256(view),
                        clean={key: value for key, value in clean.items() if not key.startswith("_")})
                    if (view.get("permutation") != artifact.get("permutation")
                            or view.get("attacks") != artifact.get("attacks")):
                        raise RuntimeError("scoring view changed the frozen layout or flips")
                    if R.model_checksum(runtime.lm) != fingerprint:
                        raise RuntimeError("model was not restored; report not published")
                    if _evidence(Path(plan["source_run"]), source, source_condition, item) != record:
                        raise RuntimeError("source evidence changed during scoring")
                    _assert_bound(run, plan, transfer_sha256, selection_sha256)
                    report = J.sanitize(report)
                    json.dumps(report, allow_nan=False)  # reject private tensors before publication
                    output.parent.mkdir(parents=True, exist_ok=True)
                    M.write_json_once(output, report)
        finally:
            runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-run", required=True)
    prepare_parser.add_argument("--source-task", choices=tuple(OPPOSITE), required=True)
    prepare_parser.add_argument("--target-run", required=True)
    prepare_parser.add_argument("--target-task", choices=tuple(OPPOSITE), required=True)
    prepare_parser.add_argument("--output", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--run", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--run", required=True)
    execute.add_argument("--condition", required=True)
    execute.add_argument("--shard", default="0/36")
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--split", choices=("dev",), required=True)
    args = parser.parse_args(argv)
    C.setup_environment()
    if args.command == "prepare":
        print(prepare(args.source_run, args.source_task, args.target_run, args.target_task, args.output))
    elif args.command == "seal":
        print(json.dumps({"sealed_items": len(seal(args.run)["records"])}))
    else:
        run_condition(args.run, args.condition, args.shard, args.device, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Publish an audited completion sentinel for one registered campaign phase.

This wrapper adds scheduling evidence without changing an existing scientific
plan or driver. Baseline/budget plans use ``campaign_protocol``; the five
separately frozen variants use ``campaign_variants``. Drivers retain their
existing per-item locks, skip behavior, legality and scientific verdicts.

``status=complete`` means that every item registered for this condition/shard
has an identity-checked evidence file. Search errors, artifact errors, failed
clean gates and invalid budgets remain visible in separate statistics; coverage
is never presented as scientific success. A missing/invalid file or a resource
exception prevents sentinel publication. Partial scientific outputs are left
intact so a later explicit invocation can use the driver's resume behavior.

An existing sentinel is accepted only after reconstructing its entire inventory
and statistics from current files, checking their SHA256 and the exact plan,
stage, condition, shard and executing wrapper identity. It is never overwritten.
VQA preflight is unsharded and must carry a successful qualification report.
No datasets, checkpoints, gradients or model objects are opened by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M


SCOPE = "regionalign_campaign_stage_v1"
SELF = "regionalign/experiments/campaign_stage.py"
VARIANTS = {"no_gradient", "static_information", "fixed_columns", "equal_exposure", "distinct_weights"}
FAMILIES = {"llm_regionalign_dev": "llm", "vqa_regionalign_dev": "vqa"}


def _runner(plan):
    variant = (plan.get("campaign_policy") or {}).get("variant")
    if variant in ("baseline", "budget"):
        name = "campaign_protocol"
    elif variant in VARIANTS:
        name = "campaign_variants"
    else:
        raise ValueError("phase wrapper requires an identified campaign variant")
    return importlib.import_module(f"regionalign.experiments.{name}")


def _backend(plan):
    try:
        family = FAMILIES[plan["scope"]]
    except KeyError as error:
        raise ValueError("unsupported campaign family") from error
    return importlib.import_module(f"regionalign.experiments.{family}")


def _registered(run, plan, backend, stage, condition_id, shard):
    condition = backend._condition(plan, condition_id)
    items = backend.planned_items(condition, plan)
    if stage == "preflight":
        if plan["scope"] != "vqa_regionalign_dev" or shard != "0/1":
            raise ValueError("preflight is available only for unsharded VQA conditions")
        if not items:
            raise ValueError("condition has no registered preflight group")
        first = items[0]
        return condition, [(backend._preflight_path(run, condition_id), {
            "matched_region": first["region"], "repeat": first["repeat"],
            "matched_arms": list(backend.ARMS)})]
    items = backend.shard_items(items, shard)
    if not items:
        raise ValueError("empty phase shard is not a registered workload")
    path_for = backend._artifact_path if stage == "search" else backend._eval_path
    return condition, [(path_for(run, condition_id, item["name"]), {
        key: item[key] for key in ("region", "repeat", "seed", "arm")}) for item in items]


def _inventory(run, plan, backend, stage, condition_id, shard, *, allow_missing=False):
    condition, registered = _registered(run, plan, backend, stage, condition_id, shard)
    policy = plan["campaign_policy"]
    shared = {"schema_version": plan["schema_version"], "scope": plan["scope"],
              "plan_sha256": plan["sha256"], "condition": condition_id,
              "campaign_scope": policy["scope"], "campaign_variant": policy["variant"],
              "campaign_policy_sha256": M.object_sha256(policy)}
    files, missing = [], []
    statuses, valid, invalid, gate_passed, gate_failed, error_files = Counter(), 0, 0, 0, 0, 0
    for raw_path, identity in registered:
        path = Path(raw_path).resolve()
        if not path.is_relative_to(run):
            raise ValueError("registered phase evidence escapes the run")
        if not path.exists():
            missing.append(str(path))
            continue
        serialized = path.read_bytes()
        value = json.loads(serialized)
        if not isinstance(value, dict):
            raise ValueError(f"phase evidence is not a JSON object: {path}")
        expected = {**shared, **identity}
        if any(value.get(key) != wanted for key, wanted in expected.items()):
            raise ValueError(f"phase evidence identity differs from registration: {path}")
        for key in ("model", "format", "task"):
            if key in value and value[key] != condition[key]:
                raise ValueError(f"phase evidence {key} differs from its condition: {path}")
        status = value.get("status", "search_error" if "error" in value else "artifact_present")
        if not isinstance(status, str):
            raise ValueError(f"phase evidence has a non-string status: {path}")
        if stage == "preflight":
            if (status != "complete" or value.get("gradient_finite") is not True
                    or any(value.get("arms", {}).get(arm, {}).get("logits_allclose") is not True
                           for arm in backend.ARMS)):
                raise ValueError(f"preflight qualification is not successful: {path}")
        has_error = ("error" in value or "artifact_error" in value
                     or status in ("artifact_error", "evaluation_error", "failed", "nonfinite"))
        error_files += int(has_error)
        statuses[status] += 1
        results = value.get("results") or {}
        if not isinstance(results, dict):
            raise ValueError(f"phase evidence results must be an object: {path}")
        for result in results.values():
            if isinstance(result, dict):
                valid += result.get("valid") is True
                invalid += result.get("valid") is False
        gate = (value.get("rearranged") or {}).get("gate")
        gate_passed += gate is True
        gate_failed += gate is False
        entry = {"path": str(path), "sha256": hashlib.sha256(serialized).hexdigest(),
                 "bytes": len(serialized), "identity": expected, "status": status,
                 "has_error": bool(has_error)}
        if stage == "evaluate":
            artifact = Path(backend._artifact_path(run, condition_id, path.name)).resolve()
            artifact_hash = M.sha256(artifact)
            if value.get("artifact_file_sha256") != artifact_hash:
                raise ValueError(f"evaluation no longer binds its source artifact: {path}")
            entry.update(source_artifact=str(artifact), source_artifact_sha256=artifact_hash)
        files.append(entry)
    if missing and not allow_missing:
        raise RuntimeError(f"phase is incomplete: {len(missing)} registered files missing; first={missing[0]}")
    statistics = {"status_counts": dict(sorted(statuses.items())), "error_files": error_files,
                  "valid_budget_results": valid, "invalid_budget_results": invalid,
                  "clean_gate_passed_files": gate_passed, "clean_gate_failed_files": gate_failed}
    return {"expected_count": len(registered), "present_count": len(files), "files": files,
            "missing": missing, "statistics": statistics}


def run_stage(run, stage, condition_id, shard, output, device="cuda:0"):
    """Resume a registered phase and atomically publish coverage evidence once."""
    started = time.monotonic()
    if stage not in ("search", "evaluate", "preflight"):
        raise ValueError("stage must be search, evaluate or preflight")
    run, output = C.output_path(run), C.output_path(output)
    if output.suffix != ".json":
        raise ValueError("phase sentinel must be a JSON file")
    source_hash = M.sha256(__file__)
    raw = M.read_json(run / "plan.json")
    runner = _runner(raw)
    plan = runner.configure_from_plan(run)
    backend = _backend(plan)
    actual_plan_hash = M.sha256(run / "plan.json")
    if plan.get("sha256") != actual_plan_hash:
        raise RuntimeError("configured campaign does not bind the current plan bytes")
    identity = {"schema_version": 1, "scope": SCOPE, "status": "complete", "run": str(run),
                "stage": stage, "condition": condition_id, "shard": shard,
                "plan_sha256": actual_plan_hash, "wrapper_source": SELF,
                "wrapper_source_sha256": source_hash,
                "campaign_variant": plan["campaign_policy"]["variant"],
                "campaign_policy_sha256": M.object_sha256(plan["campaign_policy"])}
    inventory = _inventory(run, plan, backend, stage, condition_id, shard, allow_missing=True)
    if output.exists():
        saved = M.read_json(output)
        duration = saved.get("stage_seconds") if isinstance(saved, dict) else None
        if (not isinstance(saved, dict) or any(saved.get(key) != value for key, value in identity.items())
                or saved.get("inventory") != inventory or inventory["missing"]
                or type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError("existing sentinel or its phase evidence does not match")
        return saved
    execution = "existing_registered_files"
    if inventory["missing"]:
        execution = "driver_run_or_resume"
        runner.run_stage(run, stage, condition_id, shard=shard, device=device)
    # Re-admit the exact plan after the potentially long scientific phase.
    after = runner.configure_from_plan(run)
    if (after.get("sha256") != actual_plan_hash or M.sha256(run / "plan.json") != actual_plan_hash
            or M.sha256(__file__) != source_hash):
        raise RuntimeError("plan or wrapper source changed during the phase")
    inventory = _inventory(run, after, backend, stage, condition_id, shard)
    sentinel = {**identity, "inventory": inventory, "execution": execution,
                "created_at": M.utc_now(), "stage_seconds": time.monotonic() - started,
                "completion_meaning": "all registered evidence files present and identity checked; scientific outcomes remain separate"}
    M.write_json_once(output, sentinel)
    return sentinel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", required=True, choices=("search", "evaluate", "preflight"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    C.setup_environment()
    result = run_stage(args.run, args.stage, args.condition, args.shard, args.output, args.device)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "statistics": result["inventory"]["statistics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

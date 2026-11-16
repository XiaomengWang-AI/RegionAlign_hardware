"""Per-trial trace variant of the single-injection ASR evaluators (for Fig. B).

This module is a sealed *evaluation* with **exactly the same threat model, fault
laws, success rule and legality authority** as the two base evaluators --
:mod:`regionalign.experiments.llm_injection` (uniform law) and
:mod:`regionalign.experiments.llm_injection_biased` (high-order / sign-biased
law).  It adds nothing to the threat model and relaxes nothing: the only
difference is that it *records, for every trial, that trial's outcome* -- the
per-layout NLL change ``delta_nll``, the number of kept in-range flips, and the
class (``success`` / ``no_change`` / ``other_unsuccessful`` /
``numerical_failure`` / ``crash``).

Why: the base evaluators store only per-budget aggregates (ASR, Wilson CI,
means), which are enough for a bar/point plot of expected damage but not for a
damage-exceedance (survival) curve ``P(delta_nll >= x)`` across the three
layouts.  This trace gives the raw per-trial deltas that curve needs, while the
aggregates it also reports are computed by the *same* formulas as the base
evaluators, so the two agree by construction.

Law selection
-------------
The fault law is a protocol constant, frozen in the plan field
``fault_law_kind``:

* ``uniform``  -> :func:`llm_injection.sample_fault`      (bit uniform over the code)
* ``highbias`` -> :func:`llm_injection_biased.sample_fault` (bit weighted by 2**index)

Both base modules are snapshotted into the plan's ``sources``, so drift in either
law or in the shared scoring logic stops the run.  This module only *reuses* the
base evaluators' law-agnostic helpers; it never edits them, so the frozen
evaluators and their completed runs stay byte-identical.

Independence and freezing are identical to the base evaluators: the source dev
run is verified unchanged (plan hash and every consumed artifact hash) before any
trial, and this module snapshots and re-verifies its own sources.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M
from regionalign.experiments import llm as L
from regionalign.experiments import llm_injection as J
from regionalign.experiments import llm_injection_biased as JB
from regionalign.harness import core as P
from regionalign.harness import region as R

SCOPE = "llm_injection_asr_trace_v1"
SELF = "regionalign/experiments/llm_injection_trace.py"
SUCCESS_REFERENCE = J.SUCCESS_REFERENCE
DEFAULT_KS = J.DEFAULT_KS
DEFAULT_TRIALS = J.DEFAULT_TRIALS
SOURCE_RUNTIME_FILES = J.SOURCE_RUNTIME_FILES

# The two laws this trace can run, and the frozen FAULT_LAW / sampler for each.
# Changing a law or its label is a protocol change: it needs a new plan, never an
# edit of an existing one (the base modules own these constants; we only map to
# them here).
LAWS = {
    "uniform": {"fault_law": J.FAULT_LAW, "sample_fault": J.sample_fault},
    "highbias": {"fault_law": JB.FAULT_LAW, "sample_fault": JB.sample_fault},
}


# --------------------------------------------------------------------- sources
def _own_sources():
    # Snapshot this module AND both base evaluators it reuses, so drift in either
    # law or in the shared scoring logic stops the run.
    return [SELF, J.SELF, JB.SELF, "regionalign/experiments/llm.py",
            "regionalign/harness/core.py", "regionalign/harness/region.py",
            *SOURCE_RUNTIME_FILES]


# --------------------------------------------------------------------- scoring
def _trial_class(outcome):
    """Map a finite/threshold outcome to its class label (crash handled by caller)."""
    if not outcome["finite"]:
        return "numerical_failure"
    return "success" if outcome["meets_success_threshold"] else "other_unsuccessful"


def _inject_layout(rt, rows, reference, condition, item, artifact, artifact_sha,
                   plan, ks, batch_size, log):
    """Run the plan's fault law on one frozen layout, recording every trial.

    Mirrors :func:`llm_injection._inject_layout`; the aggregates are computed by
    the same formulas, and in addition each trial's ``delta_nll``, kept-flip
    count and class are appended to ``results[str(k)]["trials_trace"]``.  A
    ``no_change`` trial (empty kept set) leaves the model unchanged, so its
    ``delta_nll`` is exactly ``0.0``; a ``crash`` records ``delta_nll = None``.
    """
    sample_fault = LAWS[plan["fault_law_kind"]]["sample_fault"]
    task = condition["task"]
    report = {"schema_version": 1, "scope": SCOPE, "condition": condition["id"],
              "region": item["region"], "repeat": item["repeat"], "arm": item["arm"],
              "seed": item["seed"], "source_plan_sha256": plan["source_plan_sha256"],
              "artifact_file_sha256": artifact_sha,
              "fault_law_kind": plan["fault_law_kind"], "fault_law": plan["fault_law"],
              "trials": plan["trials"], "evaluation_k": list(ks),
              "success_reference": SUCCESS_REFERENCE}
    reasons = L._artifact_reasons(rt, artifact, condition, item, plan["source"])
    try:
        spec = P.validate_spec(artifact.get("permutation"), condition["dims"])
        reasons.extend(R.arm_reasons(item["region"], item["arm"], spec,
                                     condition["dims"], item["seed"]))
    except (ValueError, TypeError, AttributeError) as error:
        reasons.append(str(error))
        spec = P.validate_spec({}, condition["dims"])
    if reasons:
        report.update(status="artifact_error", artifact_error="; ".join(reasons), results={})
        return report

    import torch
    permuted = not P.is_identity(spec)
    if permuted:
        P.apply_permutation(rt, spec)
    try:
        rearranged = L._task_metrics(rt, rows, task, batch_size)
        logits = rt.prefill_logits(rows[:P.LOGIT_ROWS])
        logit_ok = (bool(torch.isfinite(logits).all())
                    and torch.allclose(logits, reference["_logits"],
                                       atol=P.LOGIT_ATOL, rtol=P.LOGIT_RTOL))
        drift = (rearranged["nll"] - reference["nll"]
                 if rearranged["nll"] is not None and reference["nll"] is not None else None)
        score_ok = (task != "mmlu"
                    or abs(rearranged["accuracy"] - reference["accuracy"])
                    <= plan["source"]["mmlu_score_gate"])
        gate = (rearranged["status"] == "complete" and drift is not None
                and abs(drift) <= plan["source"]["clean_gate_nll"] and logit_ok and score_ok)
        report["rearranged"] = {"identity": not permuted, "metrics": rearranged,
                                "nll_drift": drift, "logits_allclose": logit_ok,
                                "score_gate": score_ok, "gate": gate,
                                "role": "per_layout_unfaulted_diagnostic"}
        results = {}
        for k in ks:
            classes = {"success": 0, "no_change": 0, "numerical_failure": 0,
                       "crash": 0, "other_unsuccessful": 0}
            kept_sizes, deltas, trace = [], [], []
            if not gate:
                results[str(k)] = {"k": k, "valid": False, "reasons": ["clean layout gate failed"],
                                   "trials": plan["trials"], "outcomes": classes,
                                   "trials_trace": []}
                continue
            for trial in range(plan["trials"]):
                rng = random.Random(L._hash_int(SCOPE, condition["id"], item["region"]["id"],
                                                item["arm"], item["repeat"], k, trial))
                _, kept, verdict = sample_fault(rt, item["region"], k, rng)
                kept_sizes.append(len(kept))
                if not verdict["valid"]:
                    classes["no_change"] += 1
                    # empty/illegal kept set: model is unchanged, so delta is 0.
                    trace.append({"trial": trial, "kept": len(kept),
                                  "cls": "no_change", "delta_nll": 0.0})
                    continue
                normalized = P.normalize_flips(kept)
                P.apply_flips(rt.lm, normalized)
                try:
                    faulted = L._task_metrics(rt, rows, task, batch_size)
                except Exception as error:   # noqa: BLE001 - classify, do not swallow OOM
                    P.apply_flips(rt.lm, normalized)
                    if L._is_resource_error(error):
                        raise
                    classes["crash"] += 1
                    trace.append({"trial": trial, "kept": len(kept),
                                  "cls": "crash", "delta_nll": None})
                    continue
                P.apply_flips(rt.lm, normalized)
                # AGENTS.md §7: score the fault against this layout's own clean
                # metrics; the original model stays a diagnostic only.
                outcome = J._inject_success(task, faulted, rearranged, reference)
                cls = _trial_class(outcome)
                classes[cls] += 1
                delta = outcome.get("delta_nll")
                if cls != "numerical_failure":
                    deltas.append(delta)
                trace.append({"trial": trial, "kept": len(kept), "cls": cls,
                              "delta_nll": delta,
                              "relative_score_drop": outcome.get("relative_score_drop")})
            total = plan["trials"]
            finite_deltas = [d for d in deltas if d is not None and math.isfinite(d)]
            results[str(k)] = {
                "k": k, "valid": True, "trials": total, "outcomes": classes,
                "asr": classes["success"] / total if total else None,
                "asr_ci95": J._wilson(classes["success"], total),
                "mean_kept_flips": (sum(kept_sizes) / len(kept_sizes)) if kept_sizes else None,
                "mean_delta_nll": (sum(finite_deltas) / len(finite_deltas)) if finite_deltas else None,
                "trials_trace": trace,
            }
            log(f"{item['name']} k={k}: ASR={results[str(k)]['asr']} "
                f"({classes['success']}/{total}); trace={len(trace)}")
    finally:
        if permuted:
            P.apply_permutation(rt, spec, inverse=True)
    report.update(status="complete", results=results)
    return report


# --------------------------------------------------------------- plan handling
def prepare(source_run, output, regions, law="uniform", trials=DEFAULT_TRIALS, ks=DEFAULT_KS):
    """Freeze a trace-injection plan over a subset of (condition, region).

    ``law`` selects the fault law (``uniform`` or ``highbias``); it is frozen in
    the plan and cannot be changed without a new plan.
    """
    if law not in LAWS:
        raise ValueError(f"unknown fault law {law!r}; expected one of {sorted(LAWS)}")
    source_run, source, _ = J._source_plan(source_run)
    if source.get("scope") != "llm_regionalign_dev":
        raise ValueError("injection ASR supports only llm source runs")
    output = C.output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    wanted = {}
    for entry in regions:
        condition_id, region_id = entry["condition"], entry["region"]
        condition = L._condition(source, condition_id)
        known = {region["id"] for region in condition["regions"]}
        if region_id not in known:
            raise ValueError(f"{region_id} is not a region of {condition_id}")
        wanted.setdefault(condition_id, set()).add(region_id)
    conditions = []
    for condition_id, region_ids in sorted(wanted.items()):
        condition = L._condition(source, condition_id)
        items = []
        for item in L.planned_items(condition, source):
            if item["region"]["id"] not in region_ids:
                continue
            # one noperm layout is enough: its identity is shared across repeats
            if item["arm"] == "noperm" and item["repeat"] != 0:
                continue
            artifact = L._artifact_path(source_run, condition_id, item["name"])
            if not artifact.is_file():
                raise RuntimeError(f"source artifact missing: {condition_id}/{item['name']}")
            saved = M.read_json(artifact)
            if saved.get("plan_sha256") != source["sha256"]:
                raise RuntimeError(f"source artifact does not bind the source plan: {item['name']}")
            items.append({"name": item["name"], "region_id": item["region"]["id"],
                          "repeat": item["repeat"], "arm": item["arm"],
                          "artifact_sha256": M.sha256(artifact)})
        conditions.append({"id": condition_id, "model": condition["model"],
                           "format": condition["format"], "task": condition["task"],
                           "regions": sorted(region_ids), "items": items})
    plan = {"schema_version": 1, "scope": SCOPE, "created_at": M.utc_now(),
            "source_run": str(source_run), "source_plan_sha256": source["sha256"],
            "split": "dev", "development_only": True,
            "fault_law_kind": law, "fault_law": LAWS[law]["fault_law"],
            "success_reference": SUCCESS_REFERENCE,
            "trials": int(trials), "ks": [int(k) for k in ks],
            "conditions": conditions,
            "sources": J._frozen_sources(output, _own_sources())}
    plan["sha256"] = J._plan_hash(plan)
    M.write_json_once(output / "plan.json", plan)
    return output


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if plan.get("scope") != SCOPE or plan.get("schema_version") != 1:
        raise ValueError("wrong injection protocol")
    if plan.get("fault_law_kind") not in LAWS:
        raise ValueError(f"unknown fault law {plan.get('fault_law_kind')!r}")
    if J._plan_hash(plan) != plan.get("sha256"):
        raise RuntimeError("injection plan changed since preparation")
    J._verify_sources(run, plan)
    source_run, source, _ = J._source_plan(plan["source_run"])
    if source["sha256"] != plan["source_plan_sha256"]:
        raise RuntimeError("source plan changed since preparation")
    plan["source"], plan["source_run_path"] = source, source_run
    return run, plan


def run_inject(run, condition_id, shard="0/1", device="cuda:0",
               min_free_gpu_gib=L.DEFAULT_MIN_FREE_GPU_GIB,
               batch_size=L.DEFAULT_EVAL_BATCH_SIZE, item=None, ks=None, log=print):
    run, plan = load_plan(run)
    source, source_run = plan["source"], plan["source_run_path"]
    condition_plan = J._items_for(plan, condition_id)
    condition = L._condition(source, condition_id)
    ks = [int(k) for k in (ks if ks else plan["ks"])]
    if any(k not in plan["ks"] for k in ks):
        raise ValueError(f"requested ks {ks} outside the frozen plan ks {plan['ks']}")
    index_map = {entry["name"]: entry for entry in condition_plan["items"]}
    if item is not None:
        if item not in index_map:
            raise ValueError(f"{item} not in injection plan condition {condition_id}")
        selected_entries = [index_map[item]]
    else:
        planned = {it["name"]: it for it in L.planned_items(condition, source)}
        selected = L.shard_items([planned[e["name"]] for e in condition_plan["items"]], shard)
        selected_entries = [index_map[it["name"]] for it in selected]
    output = run / "inject" / L._safe_component(condition_id)
    output.mkdir(parents=True, exist_ok=True)
    pending = [entry for entry in selected_entries
               if not J._is_complete_report(output / J._report_name(entry["name"], ks))]
    if not pending:
        return
    planned = {it["name"]: it for it in L.planned_items(condition, source)}
    rt, rows, audit, load_timings = L._load_runtime(
        source_run, source, condition, plan["split"], device, min_free_gpu_gib)
    logits = rt.prefill_logits(rows[:P.LOGIT_ROWS])
    reference = L._task_metrics(rt, rows, condition["task"], batch_size)
    reference.update({"quantization": audit, "load_timings": load_timings, "_logits": logits})
    fingerprint = R.model_checksum(rt.lm)
    log(f"{condition_id} [{plan['fault_law_kind']}]: reference dev NLL={reference['nll']}; "
        f"{len(pending)} layouts x ks={ks}")
    for entry in pending:
        item_obj = planned[entry["name"]]
        path = output / J._report_name(entry["name"], ks)
        source_artifact = L._artifact_path(source_run, condition_id, entry["name"])
        if M.sha256(source_artifact) != entry["artifact_sha256"]:
            raise RuntimeError(f"source artifact drift: {entry['name']}")
        with J._item_lock(run, condition_id, J._report_name(entry["name"], ks)):
            if J._is_complete_report(path):
                continue
            J._verify_sources(run, plan)
            artifact = M.read_json(source_artifact)
            reference_public = {k: v for k, v in reference.items() if not k.startswith("_")}
            reference_item = {**reference_public, "_logits": reference["_logits"]}
            started = time.monotonic()
            try:
                report = _inject_layout(rt, rows, reference_item, condition, item_obj, artifact,
                                        entry["artifact_sha256"], plan, ks, batch_size, log)
            except Exception as error:   # noqa: BLE001
                if L._is_resource_error(error):
                    raise
                report = {"schema_version": 1, "scope": SCOPE, "condition": condition_id,
                          "region": item_obj["region"], "repeat": item_obj["repeat"],
                          "arm": item_obj["arm"], "evaluation_k": ks,
                          "fault_law_kind": plan["fault_law_kind"],
                          "status": "injection_error", "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()[-4000:]}
            report["reference"] = reference_public
            report["seconds"] = time.monotonic() - started
            if R.model_checksum(rt.lm) != fingerprint:
                raise RuntimeError("model state changed during injection; report not published")
            if report.get("status") == "complete":
                M.write_json_once(path, report)   # canonical: a finished evaluation
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                M.write_json_once(output / f"{J._report_name(entry['name'], ks)}.error-{stamp}.json",
                                  report)
                log(f"{entry['name']} {ks}: {report.get('status')} (canonical left absent)")


def run_status(run):
    run, plan = load_plan(run)
    conditions = []
    for condition in plan["conditions"]:
        directory = run / "inject" / L._safe_component(condition["id"])
        names = [item["name"] for item in condition["items"]]
        units = len(condition["items"]) * len(plan["ks"])
        done = JB._count_done(directory, names, plan["ks"])
        errors = len(list(directory.glob("*.error-*.json"))) if directory.is_dir() else 0
        conditions.append({"id": condition["id"], "units": units, "done": done,
                           "error_files": errors})
    return {"scope": SCOPE, "plan_sha256": plan["sha256"], "development_only": True,
            "fault_law_kind": plan["fault_law_kind"], "trials": plan["trials"],
            "ks": plan["ks"], "conditions": conditions,
            "unit_total": sum(c["units"] for c in conditions),
            "done_total": sum(c["done"] for c in conditions),
            "error_file_total": sum(c["error_files"] for c in conditions)}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--source-run", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--law", default="uniform", choices=sorted(LAWS))
    pre.add_argument("--regions", required=True,
                     help="JSON list of {\"condition\":..., \"region\":...}")
    pre.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    pre.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    for name in ("run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "run":
            command.add_argument("--condition", required=True)
            command.add_argument("--shard", default="0/1")
            command.add_argument("--item", default=None,
                                 help="run exactly one layout artifact by name")
            command.add_argument("--ks", default=None,
                                 help="comma-separated budget subset of the plan's ks")
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--min-free-gpu-gib", type=float,
                                 default=L.DEFAULT_MIN_FREE_GPU_GIB)
            command.add_argument("--eval-batch-size", type=int,
                                 default=L.DEFAULT_EVAL_BATCH_SIZE)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare(args.source_run, args.output, json.loads(args.regions),
                         law=args.law, trials=args.trials,
                         ks=[int(k) for k in args.ks.split(",") if k])
        print(json.dumps({"prepared": str(output)}))
    elif args.command == "run":
        ks = [int(k) for k in args.ks.split(",") if k] if args.ks else None
        run_inject(args.run, args.condition, shard=args.shard, device=args.device,
                   min_free_gpu_gib=args.min_free_gpu_gib, batch_size=args.eval_batch_size,
                   item=args.item, ks=ks)
    elif args.command == "status":
        print(json.dumps(run_status(args.run), indent=2))


if __name__ == "__main__":
    main()

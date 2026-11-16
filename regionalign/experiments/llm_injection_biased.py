"""Single-injection ASR under a sign/high-bit-biased fault law (LLM conditions).

This is the same sealed evaluation as
:mod:`regionalign.experiments.llm_injection`, with exactly one protocol
difference: the fault law's position distribution.  The element is still drawn
uniformly over the pre-registered region's page, but the *bit* is drawn with a
weight proportional to its place value (``2**index``), so the most significant
bits -- the sign bit and its neighbours, whose flips move the weight the most --
are the most likely to be hit.  Crucially this is a *bias*, not a restriction:
every stored bit keeps positive probability, so no bit is excluded and the
region-only threat model (AGENTS.md §1, any in-range bit in the region may flip,
either direction) is preserved; only the sampling density over bits changes.
Every other rule -- per-element in-range dropping decided solely by
:func:`region.check_region_attack`, the success reference (same rearranged
unfaulted layout, AGENTS.md §7), the perplexity/accuracy thresholds, the ASR
denominator, Wilson intervals, source freezing -- is reused unchanged from the
uniform-random evaluator.

Why a separate module and a separate ``SCOPE``/plan: the fault law is a protocol
constant.  Changing it must never edit an existing plan or the uniform-random
evaluator, and must not disturb that evaluator's in-flight frozen runs, so the
biased law lives here and reuses the uniform module's law-agnostic helpers
without modifying its bytes.  The law-dependent pieces (``sample_fault`` and the
functions that carry ``SCOPE``/``FAULT_LAW``) are mirrored here; they are kept
in step with ``llm_injection`` only if the *shared* evaluation logic ever
changes (both files are frozen into this plan's sources, so any drift stops a
run).
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
from regionalign.harness import core as P
from regionalign.harness import region as R

SCOPE = "llm_injection_asr_highbias_v1"
SELF = "regionalign/experiments/llm_injection_biased.py"
# Protocol constants.  The bit weight is place-value proportional (2**index):
# every bit keeps positive probability (no per-bit restriction beyond the
# region, AGENTS.md §1) while high/sign bits dominate.  Changing the weighting
# or anything in FAULT_LAW needs a new plan, never an edit.
FAULT_LAW = {
    "positions": "weighted_by_bit_place_value_over_region_page",
    "bit_weights": "two_to_the_bit_index",
    "every_bit_positive_probability": True,
    "element_distribution": "uniform_over_region_page",
    "direction": "both_directions_depending_on_stored_value",
    "in_range": "drop_out_of_range",
    "in_range_granularity": "per_element_combined_faulted_weight",
    "no_change_counts_as_unsuccessful": True,
    "legality_authority": "region.check_region_attack",
    "trial_independence": "separate_from_search_repetitions",
    "fixed_budget_label": "conditional_on_k",
}
SUCCESS_REFERENCE = J.SUCCESS_REFERENCE
DEFAULT_KS = J.DEFAULT_KS
DEFAULT_TRIALS = J.DEFAULT_TRIALS
SOURCE_RUNTIME_FILES = J.SOURCE_RUNTIME_FILES


# --------------------------------------------------------------------- sources
def _own_sources():
    # Snapshot this module AND the uniform evaluator it reuses, so drift in
    # either the biased law or the shared logic stops the run.
    return [SELF, J.SELF, "regionalign/experiments/llm.py",
            "regionalign/harness/core.py", "regionalign/harness/region.py",
            *SOURCE_RUNTIME_FILES]


# ------------------------------------------------------------------ fault model
def _bit_weights(bits):
    """Sampling weight of each code bit, proportional to its place value (2**index).

    Every bit keeps positive probability -- the region-only threat model lets any
    stored bit flip (AGENTS.md §1) -- while the most significant bits (the sign
    bit and its neighbours), whose flips move the weight the most, are the most
    likely to be hit.
    """
    return tuple(1 << b for b in range(bits))


def _sample_bit(bits, rng):
    """Draw one bit index with place-value weighting (every index possible)."""
    return rng.choices(range(bits), weights=_bit_weights(bits), k=1)[0]


def sample_fault(rt, region, k, rng):
    """Draw one biased fault and return (sampled, kept, verdict).

    Same contract and per-element in-range dropping as
    :func:`llm_injection.sample_fault`, but the bit is drawn with place-value
    weighting (:func:`_bit_weights`) instead of uniformly; the element stays
    uniform over the region's page and every bit keeps positive probability.
    Legality of every element group and of the kept set is decided solely by
    :func:`region.check_region_attack`.
    """
    tensor, bits, lo, hi = J._region_element_range(rt, region)
    positions, guard, want = set(), 0, min(k, (hi - lo) * bits)
    while len(positions) < want and guard < 200 * k + 50:
        positions.add((rng.randrange(lo, hi), _sample_bit(bits, rng)))
        guard += 1
    sampled = [[tensor, element, bit] for element, bit in sorted(positions)]
    by_element = {}
    for flip in sampled:
        by_element.setdefault(flip[1], []).append(flip)
    kept = []
    for element in sorted(by_element):
        group = by_element[element]
        if R.check_region_attack(rt, region, group, len(group))["valid"]:
            kept.extend(group)
    verdict = (R.check_region_attack(rt, region, kept, k) if kept
               else {"valid": False, "reasons": ["no in-range flips sampled"]})
    return sampled, kept, verdict


# --------------------------------------------------------------------- scoring
def _inject_layout(rt, rows, reference, condition, item, artifact, artifact_sha,
                   plan, ks, batch_size, log):
    """Run the biased fault law on one frozen layout for budgets ``ks``.

    Mirrors :func:`llm_injection._inject_layout`; the only differences are the
    biased :func:`sample_fault` and this module's ``SCOPE``.  Scoring and
    intervals are reused from the uniform evaluator unchanged.
    """
    task = condition["task"]
    report = {"schema_version": 1, "scope": SCOPE, "condition": condition["id"],
              "region": item["region"], "repeat": item["repeat"], "arm": item["arm"],
              "seed": item["seed"], "source_plan_sha256": plan["source_plan_sha256"],
              "artifact_file_sha256": artifact_sha,
              "fault_law": plan["fault_law"], "trials": plan["trials"],
              "evaluation_k": list(ks),
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
            kept_sizes, deltas = [], []
            if not gate:
                results[str(k)] = {"k": k, "valid": False, "reasons": ["clean layout gate failed"],
                                   "trials": plan["trials"], "outcomes": classes}
                continue
            for trial in range(plan["trials"]):
                rng = random.Random(L._hash_int(SCOPE, condition["id"], item["region"]["id"],
                                                item["arm"], item["repeat"], k, trial))
                _, kept, verdict = sample_fault(rt, item["region"], k, rng)
                kept_sizes.append(len(kept))
                if not verdict["valid"]:
                    classes["no_change"] += 1
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
                    continue
                P.apply_flips(rt.lm, normalized)
                # AGENTS.md §7: score the fault against this layout's own clean
                # metrics; the original model stays a diagnostic only.
                outcome = J._inject_success(task, faulted, rearranged, reference)
                if not outcome["finite"]:
                    classes["numerical_failure"] += 1
                elif outcome["meets_success_threshold"]:
                    classes["success"] += 1
                    deltas.append(outcome.get("delta_nll"))
                else:
                    classes["other_unsuccessful"] += 1
                    deltas.append(outcome.get("delta_nll"))
            total = plan["trials"]
            finite_deltas = [d for d in deltas if d is not None and math.isfinite(d)]
            results[str(k)] = {
                "k": k, "valid": True, "trials": total, "outcomes": classes,
                "asr": classes["success"] / total if total else None,
                "asr_ci95": J._wilson(classes["success"], total),
                "mean_kept_flips": (sum(kept_sizes) / len(kept_sizes)) if kept_sizes else None,
                "mean_delta_nll": (sum(finite_deltas) / len(finite_deltas)) if finite_deltas else None,
            }
            log(f"{item['name']} k={k}: ASR={results[str(k)]['asr']} "
                f"({classes['success']}/{total})")
    finally:
        if permuted:
            P.apply_permutation(rt, spec, inverse=True)
    report.update(status="complete", results=results)
    return report


# --------------------------------------------------------------- plan handling
def prepare(source_run, output, regions, trials=DEFAULT_TRIALS, ks=DEFAULT_KS):
    """Freeze a biased-injection plan over a subset of (condition, region)."""
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
            if item["arm"] == "noperm" and item["repeat"] != 0:
                continue
            artifact = L._artifact_path(source_run, condition_id, item["name"])
            if not artifact.is_file():
                raise RuntimeError(f"source artifact missing: {condition_id}/{item['name']}")
            saved = M.read_json(artifact)
            if (saved.get("plan_sha256") != source["sha256"]):
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
            "fault_law": FAULT_LAW, "success_reference": SUCCESS_REFERENCE,
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
    log(f"{condition_id}: reference dev NLL={reference['nll']}; {len(pending)} layouts x ks={ks}")
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
                          "status": "injection_error", "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()[-4000:]}
            report["reference"] = reference_public
            report["seconds"] = time.monotonic() - started
            if R.model_checksum(rt.lm) != fingerprint:
                raise RuntimeError("model state changed during injection; report not published")
            if report.get("status") == "complete":
                M.write_json_once(path, report)
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                M.write_json_once(output / f"{J._report_name(entry['name'], ks)}.error-{stamp}.json",
                                  report)
                log(f"{entry['name']} {ks}: {report.get('status')} (canonical left absent)")


def _count_done(directory, item_names, plan_ks):
    """Completed (layout, budget) units in ``directory``.

    A unit (layout, k) is done when any complete report for that layout covers
    ``k`` in its ``evaluation_k`` -- so a single-budget report (``__k1.json``)
    and a multi-budget report (``__k1-16.json``) are both credited.  (The uniform
    evaluator's ``run_status`` counts only single-budget names; this is the same
    counting made robust to multi-budget reports, kept here so the frozen uniform
    module stays byte-identical.)
    """
    plan_ks = {int(k) for k in plan_ks}
    stems = {name[:-5] if name.endswith(".json") else name: name for name in item_names}
    covered = set()
    if directory.is_dir():
        for path in directory.glob("*.json"):
            if ".error-" in path.name or not J._is_complete_report(path):
                continue
            head = path.name[:-5].rsplit("__k", 1)[0]     # "<stem>__k<suffix>.json" -> stem
            name = stems.get(head)
            if name is None:
                continue
            try:
                report = M.read_json(path)
            except (ValueError, OSError):
                continue
            for k in report.get("evaluation_k", []):
                if int(k) in plan_ks:
                    covered.add((name, int(k)))
    return sum(1 for name in item_names for k in plan_ks if (name, k) in covered)


def run_status(run):
    run, plan = load_plan(run)
    conditions = []
    for condition in plan["conditions"]:
        directory = run / "inject" / L._safe_component(condition["id"])
        units = len(condition["items"]) * len(plan["ks"])
        done = _count_done(directory, [item["name"] for item in condition["items"]], plan["ks"])
        errors = len(list(directory.glob("*.error-*.json"))) if directory.is_dir() else 0
        conditions.append({"id": condition["id"], "units": units, "done": done,
                           "error_files": errors})
    return {"scope": SCOPE, "plan_sha256": plan["sha256"], "development_only": True,
            "trials": plan["trials"], "ks": plan["ks"], "conditions": conditions,
            "unit_total": sum(c["units"] for c in conditions),
            "done_total": sum(c["done"] for c in conditions),
            "error_file_total": sum(c["error_files"] for c in conditions)}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--source-run", required=True)
    pre.add_argument("--output", required=True)
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
                         trials=args.trials,
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

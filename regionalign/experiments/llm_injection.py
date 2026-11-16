"""Single-injection ASR for frozen RegionAlign layouts (LLM conditions).

This is a sealed *evaluation* of an already-frozen search campaign: it never
searches and never changes a source layout or its registered bits.  For each
frozen layout (one ``(region, arm, repeat)`` artifact of a source ``llm`` run)
it deploys the layout's permutation on the fault-free quantized model and then
draws independent random faults *inside the pre-registered region*, applying
each and scoring the task, to estimate an attack success rate (ASR).

Fault law (frozen in the plan, see ``FAULT_LAW``)
-------------------------------------------------
* Budget ``K`` fixed per batch of trials (conditional-on-K ASR).  A trial draws
  ``K`` distinct storage positions ``(element, bit)`` uniformly from the
  region's page; the bit is uniform over the code's bits, so both flip
  directions occur.
* In-range handling is ``drop_out_of_range``: a sampled position whose flip
  violates the threat model's in-range rule is dropped and the remaining
  in-range positions are applied.  Legality of every position and of the kept
  set is decided only by :func:`region.check_region_attack`, never by this
  module.  A trial whose kept set is empty is a no-change outcome.
* Every started trial stays in the ASR denominator.  No-change, crashes and
  numerical failures count as unsuccessful and are also reported separately.

Success reference
-----------------
Per AGENTS.md §7, fault damage is measured against *the same rearranged layout's
own* fault-free model, so each arm's own rearrangement effect is divided out and
only the fault's damage remains.  A trial succeeds when inference completes with
finite task metrics and the accuracy or VQA score drops by at least 50%, or the
negative log-likelihood rises by at least ``log 2`` (perplexity at least
doubles), against that per-layout reference.  Because every accepted layout
passes the clean gate (within ``clean_gate_nll`` of the original model), this
per-layout reference is within tolerance of the original fault-free model that
the paper's metric text calls the common reference; the difference versus the
original is also recorded as a diagnostic.

Independence
------------
These injection trials are separate from the search repetitions.  The source
run is verified unchanged (plan hash and every consumed artifact hash) before
any trial, and this module snapshots and re-verifies its own sources, so a
drift in either the source campaign or this code stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import socket
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M
from regionalign.experiments import llm as L
from regionalign.harness import core as P
from regionalign.harness import region as R

SCOPE = "llm_injection_asr_v1"
SELF = "regionalign/experiments/llm_injection.py"
# The fault law and success rule are protocol constants: they fix what an ASR
# number means and whether two runs are comparable.  Changing them needs a new
# plan, never an edit of an existing one.
FAULT_LAW = {
    "positions": "uniform_over_region_page_storage_bits",
    "direction": "uniform_bit_so_both_directions",
    "in_range": "drop_out_of_range",
    "in_range_granularity": "per_element_combined_faulted_weight",
    "no_change_counts_as_unsuccessful": True,
    "legality_authority": "region.check_region_attack",
    "trial_independence": "separate_from_search_repetitions",
    "fixed_budget_label": "conditional_on_k",
}
SUCCESS_REFERENCE = "same_rearranged_unfaulted_layout"
PPL_DOUBLING_DELTA_NLL = math.log(2.0)   # success threshold in log space (no overflow)
DEFAULT_KS = (1, 16)
DEFAULT_TRIALS = 200
SOURCE_RUNTIME_FILES = ("regionalign/runtimes/text.py",)


# --------------------------------------------------------------------- sources
def _own_sources():
    return [SELF, "regionalign/experiments/llm.py",
            "regionalign/harness/core.py", "regionalign/harness/region.py",
            *SOURCE_RUNTIME_FILES]


def _frozen_sources(run, relatives):
    destination = Path(run) / "source"
    destination.mkdir(parents=True, exist_ok=True)
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


def _verify_sources(run, plan):
    for record in plan["sources"]:
        if (M.sha256(C.PROJECT / record["path"]) != record["sha256"]
                or M.sha256(Path(run) / record["snapshot"]) != record["sha256"]):
            raise RuntimeError(f"injection source drift: {record['path']}")


# ------------------------------------------------------------------ fault model
def _region_element_range(rt, region):
    """[lo, hi) of code elements whose stored bits live on the region's page."""
    tensor = region["tensor"]
    bits = P.code_bits(rt.lm, tensor)
    if region.get("bits") != bits:
        raise RuntimeError(
            f"region declares {region.get('bits')}-bit codes but model stores {bits}-bit")
    per_page = R.elements_per_page(bits=bits, page_bytes=region["page_bytes"])
    numel = rt.lm.get_parameter(tensor).numel()
    lo = region["page"] * per_page
    hi = min(lo + per_page, numel)
    if lo >= hi:
        raise RuntimeError(f"empty region page for {tensor} page {region['page']}")
    return tensor, bits, lo, hi


def sample_fault(rt, region, k, rng):
    """Draw one random fault and return (sampled, kept, verdict).

    ``sampled`` are the ``k`` drawn positions.  The in-range rule constrains the
    *final* faulted weight of each code, so dropping is decided per *element*:
    all bits that land on one element are combined and that element is kept only
    if its combined faulted weight passes the rule (two bits that are jointly
    legal but singly illegal are kept, and vice versa).  Legality of every
    element group and of the kept set is decided solely by
    :func:`region.check_region_attack`.  ``kept`` is the union of the kept
    elements' bits; ``verdict`` is the evaluator's check of ``kept`` under ``k``.
    """
    tensor, bits, lo, hi = _region_element_range(rt, region)
    positions, guard, want = set(), 0, min(k, (hi - lo) * bits)
    while len(positions) < want and guard < 200 * k + 50:
        positions.add((rng.randrange(lo, hi), rng.randrange(bits)))
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
def _inject_success(task, faulted, reference, original_clean):
    """Success decision for one injected fault.

    ``reference`` is the same rearranged unfaulted layout (AGENTS.md §7);
    ``original_clean`` is the original fault-free model, kept only for diagnostics.
    The perplexity rule is applied in log space (``delta_nll >= log 2``) so a
    catastrophic fault whose exponentiated ratio overflows is still a success.
    """
    if faulted.get("status") != "complete":
        return {"meets_success_threshold": False, "finite": False}
    out = {"finite": True}
    delta = (faulted["nll"] - reference["nll"]
             if faulted.get("nll") is not None and reference.get("nll") is not None
             else None)
    original_delta = (faulted["nll"] - original_clean["nll"]
                      if faulted.get("nll") is not None
                      and original_clean.get("nll") is not None else None)
    out["delta_nll"] = delta
    out["ppl_ratio"] = L._ppl_ratio(delta)   # diagnostic; may be None on overflow
    out["diagnostic_delta_nll_from_original_clean"] = original_delta
    if task == "mmlu":
        base = reference.get("accuracy")
        out["relative_score_drop"] = (
            (base - faulted["accuracy"]) / base if base else None)
        original_base = original_clean.get("accuracy")
        out["diagnostic_relative_score_drop_from_original_clean"] = (
            (original_base - faulted["accuracy"]) / original_base if original_base else None)
        out["meets_success_threshold"] = bool(
            out["relative_score_drop"] is not None and out["relative_score_drop"] >= 0.5)
    else:
        out["meets_success_threshold"] = bool(
            delta is not None and math.isfinite(delta) and delta >= PPL_DOUBLING_DELTA_NLL)
    return out


def _wilson(successes, total, z=1.959963984540054):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return [None, None]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _inject_layout(rt, rows, reference, condition, item, artifact, artifact_sha,
                   plan, ks, batch_size, log):
    """Run the fault law on one frozen layout for budgets ``ks`` and report it."""
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
                outcome = _inject_success(task, faulted, rearranged, reference)
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
                "asr_ci95": _wilson(classes["success"], total),
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
    """Freeze an injection plan over a subset of (condition, region) of a source run."""
    source_run, source, module = _source_plan(source_run)
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
            "sources": _frozen_sources(output, _own_sources())}
    plan["sha256"] = _plan_hash(plan)
    M.write_json_once(output / "plan.json", plan)
    return output


def _plan_hash(plan):
    """Canonical hash of the plan with its own ``sha256`` field excluded."""
    body = {key: value for key, value in plan.items()
            if key not in ("sha256", "source", "source_run_path")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _source_plan(source_run):
    source_run = C.output_path(source_run)
    raw = M.read_json(source_run / "plan.json")
    if raw.get("campaign_policy"):
        from regionalign.experiments.campaign_protocol import configure_from_plan
        configure_from_plan(source_run)
    _, source = L.load_plan(source_run)
    return source_run, source, L


def load_plan(run):
    run = C.output_path(run)
    plan = M.read_json(run / "plan.json")
    if plan.get("scope") != SCOPE or plan.get("schema_version") != 1:
        raise ValueError("wrong injection protocol")
    if _plan_hash(plan) != plan.get("sha256"):
        raise RuntimeError("injection plan changed since preparation")
    _verify_sources(run, plan)
    source_run, source, _ = _source_plan(plan["source_run"])
    if source["sha256"] != plan["source_plan_sha256"]:
        raise RuntimeError("source plan changed since preparation")
    plan["source"], plan["source_run_path"] = source, source_run
    return run, plan


def _items_for(plan, condition_id):
    for condition in plan["conditions"]:
        if condition["id"] == condition_id:
            return condition
    raise ValueError(f"condition {condition_id!r} not in injection plan")


def _ks_suffix(ks):
    return "k" + "-".join(str(k) for k in ks)


def _report_name(item_name, ks):
    """Per-(layout, budget-set) report name, e.g. L..__r0__local__k1.json."""
    return f"{item_name[:-5] if item_name.endswith('.json') else item_name}__{_ks_suffix(ks)}.json"


def run_inject(run, condition_id, shard="0/1", device="cuda:0",
               min_free_gpu_gib=L.DEFAULT_MIN_FREE_GPU_GIB,
               batch_size=L.DEFAULT_EVAL_BATCH_SIZE, item=None, ks=None, log=print):
    run, plan = load_plan(run)
    source, source_run = plan["source"], plan["source_run_path"]
    condition_plan = _items_for(plan, condition_id)
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
    # every item is evaluated whole; only its finished reports are reused
    pending = [entry for entry in selected_entries
               if not _is_complete_report(output / _report_name(entry["name"], ks))]
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
        path = output / _report_name(entry["name"], ks)
        source_artifact = L._artifact_path(source_run, condition_id, entry["name"])
        if M.sha256(source_artifact) != entry["artifact_sha256"]:
            raise RuntimeError(f"source artifact drift: {entry['name']}")
        with _item_lock(run, condition_id, _report_name(entry["name"], ks)):
            if _is_complete_report(path):
                continue
            _verify_sources(run, plan)
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
                M.write_json_once(path, report)   # canonical: a finished evaluation
            else:
                # A failed task never occupies the canonical path, so a later run
                # can retry it; the evidence is kept under a unique name.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                M.write_json_once(output / f"{_report_name(entry['name'], ks)}.error-{stamp}.json",
                                  report)
                log(f"{entry['name']} {ks}: {report.get('status')} (canonical left absent)")


@contextmanager
def _item_lock(run, condition_id, name):
    import fcntl
    directory = Path(run) / ".locks" / "inject" / L._safe_component(condition_id)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / f"{L._safe_component(name)}.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"work item already running: {condition_id}/{name}") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _is_complete_report(path):
    """A canonical report counts as done only if it is a finished evaluation."""
    if not path.is_file():
        return False
    try:
        return M.read_json(path).get("status") == "complete"
    except (ValueError, OSError):
        return False


def run_status(run):
    run, plan = load_plan(run)
    conditions = []
    for condition in plan["conditions"]:
        directory = run / "inject" / L._safe_component(condition["id"])
        # a unit is one (layout, single budget); completion is counted per unit
        units = len(condition["items"]) * len(plan["ks"])
        done = sum(_is_complete_report(directory / _report_name(item["name"], [k]))
                   for item in condition["items"] for k in plan["ks"])
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

"""Independent, no-fault checkpoint layout validation on frozen dev panels.

This supplement reads a formal LLM or VQA plan without changing it. Its own
protocol and source snapshots are published in a new output directory before
model execution; the original plan is not claimed to contain this evaluator.
The ``run`` command gives each registered region one seeded, result-independent,
nonidentity legal row placement. Original, rearranged, and inverse-restored
layouts are measured without gradients, search, or faults.

The default uses the entire registered dev panel. ``--max-rows`` instead runs
an explicitly labelled prefix qualification and cannot establish full-panel
equivalence. ``qualify`` separately measures batch-one forward/backward resource
use on the shortest/longest frozen search rows and four registered target kinds.
It makes no parameter updates and cannot guarantee full-search capacity.
Existing loader memory gates still apply unless a campaign plan freezes a
separately identified resource admission policy.
State digests stream exact decoded parameter/scales bytes and include visual
parameters and the runtime's residual order; transient caches are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import gc
import math
import random
import shutil
import time
import traceback
from pathlib import Path

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M
from regionalign.harness import core as P, region as R

SCHEMA_VERSION = 1
SCOPE = "regionalign_layout_validation_dev"
DEFAULT_SEED = 2026111001


class RestorationError(RuntimeError):
    """Exact model/runtime state did not return after an inverse transform."""

    def __init__(self, message, expected=None, actual=None):
        super().__init__(message)
        self.expected, self.actual = expected, actual


def _adapter(family):
    if family == "llm":
        from regionalign.experiments import llm
        return llm
    if family == "vqa":
        from regionalign.experiments import vqa
        return vqa
    raise ValueError("family must be llm or vqa")


def _configure_campaign(run):
    """Apply only the supplemental policy explicitly frozen in a campaign plan."""
    plan = M.read_json(C.output_path(run) / "plan.json")
    if "campaign_policy" in plan:
        from regionalign.experiments import campaign_protocol
        return campaign_protocol.configure_from_plan(run)["campaign_policy"]
    return None


def planned_layouts(condition, seed, region_ids=None):
    """Freeze one nonidentity placement per region using only IDs/dimensions."""
    wanted = set(region_ids) if region_ids is not None else None
    regions = condition["regions"]
    if wanted is not None and (not wanted or wanted - {r["id"] for r in regions}):
        raise ValueError("region IDs must select registered regions")
    layouts = []
    for region in regions:
        if wanted is not None and region["id"] not in wanted:
            continue
        target = R.primary_segment(region, condition["dims"])[0]
        choices = [row for row in R.reachable_rows(region, condition["dims"])
                   if row != target]
        derived = hashlib.sha256(
            f"{seed}|{condition['id']}|{region['id']}".encode()).hexdigest()
        source = random.Random(int(derived[:16], 16)).choice(choices) if choices else None
        spec = R.placement(region, condition["dims"], source) if source is not None else None
        if spec is not None and P.is_identity(spec):
            raise ValueError("nonidentity layout policy produced an identity transform")
        layouts.append({"region": region, "source_row": source,
                        "selection_seed_sha256": derived,
                        "permutation": P.spec_to_json(spec) if spec is not None else None,
                        "applicable": spec is not None})
    if not layouts:
        raise ValueError("no registered regions selected")
    return layouts


def runtime_state(runtime):
    """Exact byte digest including language scales, excluded vision, and order."""
    language = M.quantized_state_manifest(runtime.lm)
    language_ids = {id(p) for p in runtime.lm.parameters()}
    model = getattr(runtime, "model", runtime.lm)
    visual = [M._tensor_record(name, value) for name, value in sorted(model.named_parameters())
              if id(value) not in language_ids]
    residual = getattr(runtime, "residual", None)
    residual = M._tensor_record("runtime.residual", residual) if residual is not None else None
    state = {"language": language, "visual_parameters": visual, "residual": residual}
    return {"algorithm": "sha256_exact_tensor_bytes_and_runtime_order_v1",
            "sha256": M.object_sha256(state),
            "language_manifest_sha256": language["manifest_sha256"],
            "visual_parameters_sha256": M.object_sha256(visual),
            "residual": residual}


def logits_comparison(logits, reference):
    import torch
    value = logits.detach().float().contiguous().cpu()
    finite = bool(torch.isfinite(value).all())
    same_shape = list(value.shape) == list(reference.shape)
    reference_finite = bool(torch.isfinite(reference).all())
    comparable = finite and reference_finite and same_shape
    difference = (value - reference).abs() if comparable else None
    return {"shape": list(value.shape), "finite": finite,
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "allclose": bool(comparable and torch.allclose(
                value, reference, atol=P.LOGIT_ATOL, rtol=P.LOGIT_RTOL)),
            "max_abs_difference": float(difference.max()) if comparable else None,
            "atol": P.LOGIT_ATOL, "rtol": P.LOGIT_RTOL}


def metric_gate(metrics, original, task, plan):
    """Apply the originating plan's clean gates without changing their meaning."""
    nll, baseline = metrics.get("nll"), original.get("nll")
    finite = all(isinstance(v, (int, float)) and math.isfinite(v)
                 for v in (nll, baseline))
    drift = nll - baseline if finite else None
    score_key = "accuracy" if task == "mmlu" else "score" if task in ("okvqa", "textvqa") else None
    score_drift = None
    score_ok = True
    if score_key is not None:
        a, b = metrics.get(score_key), original.get(score_key)
        ok = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b))
        score_drift = a - b if ok else None
        tolerance = plan["mmlu_score_gate" if task == "mmlu" else "vqa_score_gate"]
        score_ok = ok and abs(score_drift) <= tolerance
    return {"nll_drift": drift, "score_drift": score_drift,
            "score_gate": bool(score_ok),
            "metrics_gate": bool(metrics.get("status") == "complete" and finite
                                 and abs(drift) <= plan["clean_gate_nll"] and score_ok)}


def validate_layout(runtime, rows, condition, plan, layout, original, reference, state, measure):
    """Evaluate one frozen layout; inverse restoration also runs after failures."""
    if not layout["applicable"]:
        return {**layout, "status": "not_applicable", "reason": "no legal nonidentity reachable row"}
    spec = P.validate_spec(layout["permutation"], condition["dims"])
    report = {**layout, "status": "running"}
    applied = False
    try:
        P.apply_permutation(runtime, spec)
        applied = True
        metrics = measure(runtime, rows)
        logits = logits_comparison(runtime.prefill_logits(rows[:P.LOGIT_ROWS]), reference)
        gate = metric_gate(metrics, original, condition["task"], plan)
        report["rearranged"] = {"metrics": metrics, "logits": logits,
                                "state": runtime_state(runtime), **gate,
                                "gate": gate["metrics_gate"] and logits["allclose"]}
    finally:
        if applied:
            P.apply_permutation(runtime, spec, inverse=True)
            restored = runtime_state(runtime)
            report["inverse_state"] = restored
            if restored != state:
                raise RestorationError("inverse permutation did not restore exact runtime state", state, restored)
        else:
            current = runtime_state(runtime)
            if current != state:
                raise RestorationError("incomplete layout application changed model state; stopping", state, current)
    metrics = measure(runtime, rows)
    logits = logits_comparison(runtime.prefill_logits(rows[:P.LOGIT_ROWS]), reference)
    gate = metric_gate(metrics, original, condition["task"], plan)
    report["restored"] = {"metrics": metrics, "logits": logits, **gate,
                           "state_matches_original": True,
                           "gate": gate["metrics_gate"] and logits["allclose"]}
    report["status"] = ("complete" if report["rearranged"]["gate"] and report["restored"]["gate"]
                        else "equivalence_failed")
    return report


def _failure_status(error):
    if isinstance(error, RestorationError):
        return "restoration_failed"
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if isinstance(current, MemoryError) or any(word in text for word in (
                "out of memory", "outofmemory", "cannot allocate memory", "cublas_status_alloc_failed",
                "insufficient free gpu memory", "requires at least", "requires 40 gib",
                "no space left", "disk quota")):
            return "resource_error"
        current = current.__cause__ or current.__context__
    message = str(error).lower()
    if "original no-fault metrics or logits are nonfinite" in message:
        return "nonfinite_baseline"
    if "is unavailable:" in message:
        return "unavailable"
    protocol = isinstance(error, (ValueError, FileNotFoundError)) or any(token in message for token in (
        "drift", "sources changed", "source changed", "does not register", "does not bind"))
    return "protocol_error" if protocol else "execution_error"


def qualification_probes(runtime, rows, condition):
    """Select shortest/longest search rows and the earliest registered target per kind.

    Length includes answer tokens and, for VLMs, the processor's image-expanded
    prompt tokens. Ties use SID. This is a fixed resource stress sample, never
    an estimate of task quality or a guarantee about all subsequent workloads.
    """
    lengths = []
    for row in rows:
        if "prompt_ids" in row:
            length = len(row["prompt_ids"]) + len(row["answer_ids"])
        else:
            length = int(runtime._prompt(row)["input_ids"].shape[-1]) + row["target_tokens"]
        lengths.append((length, row["sid"], row))
    if not lengths:
        raise ValueError("search qualification panel is empty")
    ordered = sorted(lengths, key=lambda item: (item[0], item[1]))
    probes = []
    targets = {}
    for region in sorted(condition["regions"], key=lambda value: (value["layer"], value["id"])):
        targets.setdefault(region["kind"], region["tensor"])
    if set(targets) != {"q_proj", "v_proj", "gate_proj", "down_proj"}:
        raise ValueError("qualification requires the four registered target kinds")
    for role, (length, sid, row) in (("shortest", ordered[0]), ("longest", ordered[-1])):
        for kind, name in sorted(targets.items()):
            probes.append({"role": role, "sid": sid, "input_tokens": length,
                           "kind": kind, "tensor": name, "row": row})
    return probes


def qualify_runtime(runtime, rows, condition, device, output):
    """Measure unmodified batch-one gradients; save no gradient values or faults."""
    import torch
    probes = qualification_probes(runtime, rows, condition)
    selection = [{key: value for key, value in probe.items() if key != "row"} for probe in probes]
    M.write_json_once(output / "qualification_selection.json", {
        "selection_rule": "shortest_longest_search_rows_and_earliest_registered_tensor_per_kind_v1",
        "split": "search", "batch_size": 1, "targets_per_row": 4,
        "probes": selection, "scientific_result": False})
    before = runtime_state(runtime)
    records = []
    cuda = str(device).startswith("cuda")
    for index, probe in enumerate(probes):
        record = {key: value for key, value in probe.items() if key != "row"}
        gradient = gradients = None
        started = time.monotonic()
        try:
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                record["allocated_before_bytes"] = torch.cuda.memory_allocated(device)
            values = runtime.nll([probe["row"]], bs=1)
            record["forward_finite"] = len(values) == 1 and math.isfinite(values[0])
            record["forward_nll"] = P.reportable(values[0]) if len(values) == 1 else None
            gradients = runtime.grads([probe["tensor"]], [probe["row"]], bs=1)
            gradient = gradients[probe["tensor"]]
            if gradient is None:
                raise RuntimeError("requested target has no gradient")
            record.update(gradient_shape=list(gradient.shape), gradient_dtype=str(gradient.dtype),
                          gradient_elements=gradient.numel(),
                          gradient_finite=bool(torch.isfinite(gradient).all()))
            if cuda:
                torch.cuda.synchronize(device)
            record["status"] = ("complete" if record["forward_finite"] and record["gradient_finite"]
                                else "nonfinite")
        except Exception as error:
            record.update(status=_failure_status(error), error_type=type(error).__name__, error=str(error))
        finally:
            if cuda:
                record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                record["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            record["seconds"] = time.monotonic() - started
            del gradient, gradients
            runtime.lm.zero_grad(set_to_none=True)
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        M.write_json_once(output / f"gradient_probe_{index:02d}.json", record)
        records.append(record)
        if record["status"] not in ("complete", "nonfinite", "resource_error"):
            break
    after = runtime_state(runtime)
    if before != after:
        raise RestorationError("resource qualification changed exact model/runtime state", before, after)
    status = ("complete" if len(records) == len(probes) and all(r["status"] == "complete" for r in records)
              else "resource_error" if any(r["status"] == "resource_error" for r in records)
              else "qualification_failed")
    return {"status": status, "scientific_result": False, "split": "search",
            "batch_size": 1, "registered_probes": len(probes), "attempted_probes": len(records),
            "completed_probes": sum(r["status"] == "complete" for r in records),
            "probes": records, "before_state": before, "after_state": after,
            "state_matches_original": True,
            "qualification_scope": "two_frozen_search_rows_four_registered_target_kinds_no_faults",
            "covers_all_search_rows": False, "covers_full_search_algorithm": False,
            "peak_gpu_allocated_bytes": max((r.get("peak_gpu_allocated_bytes", 0) for r in records), default=0) if cuda else None,
            "peak_gpu_reserved_bytes": max((r.get("peak_gpu_reserved_bytes", 0) for r in records), default=0) if cuda else None}


def _snapshot_sources(output, plan=None):
    paths = {Path(__file__).resolve().relative_to(C.PROJECT)}
    paths.update(Path(record["path"]) for record in (plan or {}).get("sources", []))
    records = []
    for relative in sorted(paths):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source inventory path")
        source = C.PROJECT / relative
        target = output / "sources" / str(relative).replace("/", "__")
        if target.exists():
            continue
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        record = M.file_record(source, C.PROJECT)
        if M.sha256(target) != record["sha256"]:
            raise RuntimeError("source changed during supplemental snapshot")
        records.append({**record, "snapshot": str(target.relative_to(output))})
    return records


def run_validation(run, condition_id, *, family="llm", output=None, device="cuda:0",
                   seed=DEFAULT_SEED, max_rows=None, region_ids=None, batch_size=1,
                   min_free_gpu_gib=None, operation="validate"):
    """Write a new independent report, including classified execution failures."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be positive")
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ValueError("max rows must be positive")
    if min_free_gpu_gib is not None and (not math.isfinite(min_free_gpu_gib) or min_free_gpu_gib < 0):
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    if operation not in ("validate", "qualify"):
        raise ValueError("operation must be validate or qualify")
    if operation == "qualify" and (max_rows is not None or region_ids is not None or batch_size != 1):
        raise ValueError("qualification uses its frozen row/target rule and batch size one")
    split = "search" if operation == "qualify" else "dev"
    destination = C.fresh_output_dir("layout_validation", condition_id, output)
    started = time.monotonic()
    report = {"schema_version": SCHEMA_VERSION, "scope": SCOPE,
              "created_at": M.utc_now(), "condition": condition_id, "family": family,
              "development_only": True, "split": split, "no_faults_or_search": True,
              "operation": operation, "gradient_qualification": "not_performed", "cases": [],
              "original_plan_contains_this_evaluator": False,
              "sources": [], "status": "running"}
    runtime = None
    try:
        report["sources"] = _snapshot_sources(destination)
        C.setup_environment()
        report["campaign_policy"] = _configure_campaign(run)
        backend = _adapter(family)
        run, plan = backend.load_plan(run)
        report.update(plan_run=str(run), plan_sha256=plan["sha256"], plan_scope=plan["scope"])
        report["sources"].extend(_snapshot_sources(destination, plan))
        condition = backend._condition(plan, condition_id)
        layouts = planned_layouts(condition, seed, region_ids) if operation == "validate" else []
        memory = (min_free_gpu_gib if min_free_gpu_gib is not None else
                  plan["execution_protocol"]["min_free_gpu_gib"] if family == "llm" else
                  plan["execution"]["min_free_gpu_gib"])
        if family == "vqa":
            from regionalign.experiments import clean_benchmarks
            memory = max(memory, plan["execution"]["min_free_gpu_gib"],
                         clean_benchmarks.SETTINGS["min_free_gpu_gib"])
        protocol = {"scope": SCOPE, "condition": condition_id, "family": family,
                    "plan_sha256": plan["sha256"], "split": split, "seed": seed,
                    "operation": operation, "campaign_policy": report["campaign_policy"],
                    "layout_policy": "seeded_nonidentity_reachable_row_v1", "layouts": layouts,
                    "max_rows": max_rows, "batch_size": batch_size, "device": device,
                    "requested_min_free_gpu_gib": min_free_gpu_gib,
                    "effective_min_free_gpu_gib": memory, "sources": report["sources"],
                    "clean_gate_nll": plan["clean_gate_nll"],
                    "logit_atol": P.LOGIT_ATOL, "logit_rtol": P.LOGIT_RTOL,
                    "resource_qualification_scope": ("inference_and_layout_only" if operation == "validate"
                                                      else "batch_one_forward_and_target_gradient_only")}
        M.write_json_once(destination / "protocol.json", protocol)
        report["protocol_sha256"] = M.sha256(destination / "protocol.json")
        import torch
        if str(device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info(device)
            report["resources_before_load"] = {"free_bytes": free, "total_bytes": total}
            if free < int(memory * 1024 ** 3):
                raise RuntimeError(f"insufficient free GPU memory: {free} < {int(memory * 1024 ** 3)}")
            torch.cuda.reset_peak_memory_stats(device)
        report["device"] = M.runtime_device_identity(torch, device)
        if family == "llm":
            runtime, frozen_rows, audit, timing = backend._load_runtime(
                run, plan, condition, split, device, memory)
            measure = lambda rt, rs: backend._task_metrics(rt, rs, condition["task"], batch_size)
        else:
            if batch_size != backend.EVAL_BATCH_SIZE:
                raise ValueError("VQA batch size must match its runtime evaluation batch size")
            runtime, frozen_rows = backend._load_runtime(run, plan, condition, split, device)
            audit, timing = runtime.quantization, runtime.load_timing
            measure = backend._task_metrics
        report["environment"] = M.software_environment()
        if str(device).startswith("cuda"):
            report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["load_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        rows = frozen_rows[:max_rows] if max_rows is not None else frozen_rows
        if not rows:
            raise ValueError("registered dev panel is empty")
        report.update(registered_panel_rows=len(frozen_rows), evaluated_rows=len(rows),
                      panel_coverage="full" if len(rows) == len(frozen_rows) else "prefix_qualification",
                      row_ids_sha256=M.object_sha256([r["sid"] for r in rows]),
                      quantization=audit, load_timing=timing)
        if operation == "qualify":
            report["qualification"] = qualify_runtime(runtime, rows, condition, device, destination)
            selected = sorted({probe["sid"] for probe in report["qualification"]["probes"]})
            report["frozen_search_population_row_ids_sha256"] = report["row_ids_sha256"]
            report.update(evaluated_rows=len(selected), row_ids_sha256=M.object_sha256(selected),
                          panel_coverage="shortest_longest_search_qualification")
            report["gradient_qualification"] = report["qualification"]["status"]
            report["status"] = report["qualification"]["status"]
            report["scientific_result"] = False
        else:
            _run_layout_cases(runtime, rows, condition, plan, layouts, measure, report, destination)
        backend._assert_frozen(run, plan)
        for source in report["sources"]:
            if M.sha256(C.PROJECT / source["path"]) != source["sha256"]:
                raise RuntimeError("supplement source changed during validation")
    except Exception as error:
        report.update(status=_failure_status(error), error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc()[-6000:])
        if isinstance(error, RestorationError):
            report.update(expected_state=error.expected, actual_state=error.actual)
    finally:
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception as error:
                report.update(status=_failure_status(error), close_error=str(error))
        try:
            import torch
            if str(device).startswith("cuda") and torch.cuda.is_available():
                report["peak_gpu_allocated_bytes"] = max(
                    torch.cuda.max_memory_allocated(device), report.get("load_peak_gpu_allocated_bytes", 0),
                    report.get("qualification", {}).get("peak_gpu_allocated_bytes") or 0)
                report["peak_gpu_reserved_bytes"] = max(
                    torch.cuda.max_memory_reserved(device), report.get("load_peak_gpu_reserved_bytes", 0),
                    report.get("qualification", {}).get("peak_gpu_reserved_bytes") or 0)
        except Exception:
            pass
        report.update(completed_at=M.utc_now(), seconds=time.monotonic() - started)
        M.write_json_once(destination / "report.json", report)
    return destination, report


def _run_layout_cases(runtime, rows, condition, plan, layouts, measure, report, destination):
    state = runtime_state(runtime)
    original = measure(runtime, rows)
    reference = runtime.prefill_logits(rows[:P.LOGIT_ROWS]).detach().float().cpu()
    report["original"] = {"metrics": original, "state": state,
                          "logits": logits_comparison(reference, reference)}
    M.write_json_once(destination / "original.json", report["original"])
    if original.get("status") != "complete" or not report["original"]["logits"]["finite"]:
        raise RuntimeError("original no-fault metrics or logits are nonfinite")
    (destination / "cases").mkdir()
    for index, layout in enumerate(layouts):
        report["active_layout"] = {"index": index, "region_id": layout["region"]["id"]}
        case = validate_layout(runtime, rows, condition, plan, layout, original,
                               reference, state, measure)
        path = destination / "cases" / f"{index:02d}.json"
        M.write_json_once(path, case)
        report["cases"].append({"region_id": layout["region"]["id"],
                                "status": case["status"], "path": str(path.relative_to(destination)),
                                "sha256": M.sha256(path)})
    report.pop("active_layout", None)
    applicable = [case for case in report["cases"] if case["status"] != "not_applicable"]
    report["status"] = ("complete" if applicable and all(case["status"] == "complete" for case in applicable)
                        else "equivalence_failed" if applicable else "not_applicable")
    report["inference_resource_qualification"] = "passed"


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "qualify"):
        run = commands.add_parser(name, help="no-fault dev equivalence" if name == "run" else "search-panel gradient resource qualification only")
        run.add_argument("--family", choices=("llm", "vqa"), required=True)
        run.add_argument("--run", required=True, help="existing formal development plan directory")
        run.add_argument("--condition", required=True)
        run.add_argument("--output", help="new directory under the result root")
        run.add_argument("--device", default="cuda:0")
        run.add_argument("--min-free-gpu-gib", type=float)
        if name == "run":
            run.add_argument("--seed", type=int, default=DEFAULT_SEED)
            run.add_argument("--max-rows", type=int)
            run.add_argument("--region-id", action="append")
            run.add_argument("--batch-size", type=int, default=1)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    destination, report = run_validation(
        args.run, args.condition, family=args.family, output=args.output, device=args.device,
        seed=getattr(args, "seed", DEFAULT_SEED), max_rows=getattr(args, "max_rows", None),
        region_ids=getattr(args, "region_id", None), batch_size=getattr(args, "batch_size", 1),
        min_free_gpu_gib=args.min_free_gpu_gib,
        operation="qualify" if args.command == "qualify" else "validate")
    print(destination, flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fork frozen development panels into separately identified campaign runs.

``prepare`` never selects data: registered panel files are copied byte for byte,
while checkpoint/data/bootstrap identities are inherited.  Current executable
sources and this module are frozen under a new plan hash.  The original formal
``scope`` remains a loader compatibility field; ``campaign_policy.scope`` and
the corresponding artifact fields identify this distinct protocol extension.

``configure_from_plan`` applies only a recorded memory admission threshold to
the current process.  It does not change quantization, data, scoring, batching,
legality or clean gates.  Use one independent process per campaign plan.  A
lower threshold records a candidate resource policy, not proof that a workload
fits: resource qualification must precede actual worker launches.

This stable module implements baseline and independent 50/100/200/300 compute
budget plans.  Additional variants belong in a separately frozen module so
adding them cannot change the identity of an already running baseline plan.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import importlib
import math
from pathlib import Path
import shutil
from unittest.mock import patch

from regionalign import config as C
from regionalign.experiments import clean_mmlu as M

CAMPAIGN_SCOPE = "regionalign_campaign_dev"
POLICY_VERSION = 1
MODULE_SOURCE = Path("regionalign/experiments/campaign_protocol.py")
FAMILIES = {"llm_regionalign_dev": "llm", "vqa_regionalign_dev": "vqa"}
VARIANTS = ("baseline", "budget")
BUDGETS = (50, 100, 200, 300)
_configuration = None
_configuration_stack = None


def _backend(scope):
    try:
        family = FAMILIES[scope]
    except KeyError as error:
        raise ValueError(f"unsupported parent scope {scope!r}") from error
    return importlib.import_module(f"regionalign.experiments.{family}")


def _memory_threshold(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("minimum free GPU memory must be finite and non-negative")
    return float(value)


def _read_registered_plan(run):
    run = Path(run).expanduser().resolve()
    plan = M.read_json(run / "plan.json")
    digest = M.sha256(run / "plan.json")
    status = M.read_json(run / "status.json")
    if status.get("plan_sha256") != digest:
        raise ValueError("status.json does not bind this plan hash")
    if (plan.get("schema_version") != 2 or plan.get("scope") not in FAMILIES
            or plan.get("development_only") is not True
            or plan.get("splits") != ["search", "dev"]):
        raise ValueError("a schema-2 formal search/dev plan is required")
    return run, plan, digest


def _saved_source_check(run, plan):
    """Check historical snapshots without requiring historical live sources."""
    if not plan.get("sources"):
        raise ValueError("parent has no registered source snapshots")
    for record in plan["sources"]:
        relative = Path(record["snapshot"])
        saved = (run / relative).resolve()
        if relative.is_absolute() or run not in saved.parents:
            raise ValueError("source snapshot escapes the parent run")
        if not saved.is_file() or M.sha256(saved) != record["sha256"]:
            raise ValueError(f"parent source snapshot changed: {relative}")


def _registered_panels(run, plan):
    records = {}
    for condition in plan["conditions"]:
        panels = condition.get("panels")
        if panels is None:
            if condition.get("availability") != "unavailable":
                raise ValueError("an available condition lacks frozen panels")
            continue
        if set(panels) != {"search", "dev"}:
            raise ValueError("only frozen search/dev panels may be inherited")
        for split, record in panels.items():
            relative = Path(record["path"])
            source = (run / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != "panels"
                    or run not in source.parents):
                raise ValueError("panel path escapes the registered panel tree")
            if "test" in relative.name.lower() or split not in ("search", "dev"):
                raise ValueError("final/test inputs cannot be inherited")
            if not source.is_file() or M.sha256(source) != record["sha256"]:
                raise ValueError(f"frozen panel changed: {relative}")
            previous = records.get(str(relative))
            if previous and previous["sha256"] != record["sha256"]:
                raise ValueError("conditions disagree about a shared panel identity")
            records[str(relative)] = {"path": str(relative), "sha256": record["sha256"],
                                      "bytes": source.stat().st_size}
    if not records:
        raise ValueError("parent registers no available frozen panels")
    return records


def prepare(parent_run, output, *, variant="baseline", min_free_gpu_gib=40.0,
            budget=300, note=""):
    """Create a new plan from preserved panels; never modify the parent run."""
    if variant not in VARIANTS or type(budget) is not int or budget not in BUDGETS:
        raise ValueError("unsupported variant or compute budget")
    if variant == "baseline" and budget != 300:
        raise ValueError("a changed compute budget requires variant='budget'")
    minimum = _memory_threshold(min_free_gpu_gib)
    parent, original, parent_hash = _read_registered_plan(parent_run)
    _saved_source_check(parent, original)
    panel_records = _registered_panels(parent, original)
    backend = _backend(original["scope"])
    output = C.output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = copy.deepcopy(original)
    plan.pop("sha256", None)
    plan.pop("campaign_policy", None)
    sources = backend._copy_sources(output)
    source = C.PROJECT / MODULE_SOURCE
    saved = output / "source" / str(MODULE_SOURCE).replace("/", "__")
    shutil.copyfile(source, saved)
    digest = M.sha256(source)
    if M.sha256(saved) != digest:
        raise RuntimeError("campaign module changed while freezing sources")
    sources.append({"path": str(MODULE_SOURCE), "snapshot": str(saved.relative_to(output)),
                    "sha256": digest, "bytes": source.stat().st_size})
    for relative, record in panel_records.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / relative, target)
        if M.sha256(target) != record["sha256"]:
            raise RuntimeError(f"inherited panel copy changed: {relative}")
    if plan["scope"] == "llm_regionalign_dev":
        execution = copy.deepcopy(plan["execution_protocol"])
        execution["min_free_gpu_gib"] = minimum
        plan["execution_protocol"] = execution
    else:
        execution = copy.deepcopy(plan["execution"])
        execution["min_free_gpu_gib"] = minimum
        plan["execution"] = execution
    plan.update(created_at=M.utc_now(), sources=sources, search_budget=budget, note=note)
    plan["campaign_policy"] = {
        "schema_version": POLICY_VERSION, "scope": CAMPAIGN_SCOPE,
        "formal_loader_scope": original["scope"], "variant": variant,
        "parameters": {"search_budget": budget},
        "parent_run": str(parent), "parent_plan_sha256": parent_hash,
        "panel_inheritance": "registered_files_copied_byte_for_byte_no_resampling",
        "inherited_panels": list(panel_records.values()),
        "resource_policy": {"min_free_gpu_gib": minimum,
                            "mode": "process_local_explicit_admission_threshold",
                            "qualification": "required_before_workload_launch"},
    }
    for record in sources:
        if M.sha256(C.PROJECT / record["path"]) != record["sha256"]:
            raise RuntimeError("source changed while preparing campaign; create a new run")
    M.write_json_once(output / "plan.json", plan)
    available = [c for c in plan["conditions"] if c.get("availability", "available") == "available"]
    M.write_json_once(output / "status.json", {
        "status": "planned", "development_only": True,
        "scope": CAMPAIGN_SCOPE, "variant": variant,
        "conditions": len(plan["conditions"]), "available_conditions": len(available),
        "unavailable_conditions": len(plan["conditions"]) - len(available),
        "registered_searches": len(available) * plan["work_items_per_condition"],
        "plan_sha256": M.sha256(output / "plan.json"),
        "parent_plan_sha256": parent_hash,
    })
    return output


def _validate_policy(run, plan):
    policy = plan.get("campaign_policy")
    if (not isinstance(policy, dict) or policy.get("scope") != CAMPAIGN_SCOPE
            or policy.get("schema_version") != POLICY_VERSION
            or policy.get("formal_loader_scope") != plan["scope"]
            or policy.get("variant") not in VARIANTS):
        raise ValueError("missing or unsupported campaign identity")
    budget = policy.get("parameters", {}).get("search_budget")
    if (type(budget) is not int or budget not in BUDGETS
            or plan.get("search_budget") != budget
            or (policy["variant"] == "baseline" and budget != 300)):
        raise ValueError("campaign budget differs from its registered variant")
    threshold = _memory_threshold(policy.get("resource_policy", {}).get("min_free_gpu_gib"))
    execution = plan["execution_protocol"] if plan["scope"] == "llm_regionalign_dev" else plan["execution"]
    if execution.get("min_free_gpu_gib") != threshold:
        raise ValueError("campaign resource policy differs from the formal execution record")
    module_records = [s for s in plan["sources"] if s["path"] == str(MODULE_SOURCE)]
    if len(module_records) != 1 or M.sha256(__file__) != module_records[0]["sha256"]:
        raise ValueError("the executing campaign module is not the frozen module")
    _saved_source_check(run, plan)
    inherited = _registered_panels(run, plan)
    if list(inherited.values()) != policy.get("inherited_panels"):
        raise ValueError("campaign panel inheritance identity changed")
    return policy


def configure_from_plan(run):
    """Configure this independent process and return the validated formal plan.

    Call before a formal driver's ``load_plan`` or ``_load_runtime``.  Repeated
    calls for the same hash are harmless; another plan requires a new process.
    Callers freezing their own protocols must include this module's source SHA.
    """
    global _configuration, _configuration_stack
    run, plan, digest = _read_registered_plan(run)
    policy = _validate_policy(run, plan)
    if _configuration is not None:
        if _configuration["sha256"] != digest:
            raise RuntimeError("a different campaign plan requires an independent process")
        return copy.deepcopy(_configuration)
    backend = _backend(plan["scope"])
    stack = ExitStack()
    try:
        if plan["scope"] == "vqa_regionalign_dev":
            from regionalign.experiments import clean_benchmarks as clean
            minimum = policy["resource_policy"]["min_free_gpu_gib"]
            stack.enter_context(patch.object(backend, "MIN_FREE_GPU_GIB", minimum))
            stack.enter_context(patch.dict(clean.SETTINGS, {"min_free_gpu_gib": minimum}))
        # Reuse the original loader's source, status, execution and split checks.
        _, loaded = backend.load_plan(run)
        original_writer = M.write_json_once
        def campaign_writer(path, value):
            if (isinstance(value, dict) and value.get("scope") == plan["scope"]
                    and value.get("plan_sha256") == digest):
                value = {**value, "campaign_scope": CAMPAIGN_SCOPE,
                         "campaign_variant": policy["variant"],
                         "campaign_policy_sha256": M.object_sha256(policy)}
            return original_writer(path, value)
        stack.enter_context(patch.object(M, "write_json_once", campaign_writer))
        _configuration_stack, _configuration = stack, copy.deepcopy(loaded)
        return copy.deepcopy(loaded)
    except BaseException:
        stack.close()
        raise


def _reset_configuration():
    """Restore process state for CPU tests; workers use a fresh process instead."""
    global _configuration, _configuration_stack
    if _configuration_stack is not None:
        _configuration_stack.close()
    _configuration = _configuration_stack = None


def run_stage(run, stage, condition, *, shard="0/1", device="cuda:0"):
    plan = configure_from_plan(run)
    backend = _backend(plan["scope"])
    if stage == "preflight":
        if plan["scope"] != "vqa_regionalign_dev":
            raise ValueError("the LLM driver has no preflight stage; use layout qualification")
        return backend.run_preflight(run, condition, device)
    if stage not in ("search", "evaluate"):
        raise ValueError("stage must be search, evaluate or preflight")
    function = backend.run_search if stage == "search" else backend.run_evaluate
    if plan["scope"] == "vqa_regionalign_dev":
        return function(run, condition, shard, device)
    execution = plan["execution_protocol"]
    kwargs = {"min_free_gpu_gib": execution["min_free_gpu_gib"],
              "batch_size": execution["search_batch_size" if stage == "search"
                                      else "evaluation_batch_size"]}
    if stage == "search":
        kwargs["grad_batch_size"] = execution["grad_batch_size"]
    return function(run, condition, shard, device, **kwargs)


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--parent-run", required=True)
    prepare_command.add_argument("--output", required=True)
    prepare_command.add_argument("--variant", choices=VARIANTS, default="baseline")
    prepare_command.add_argument("--min-free-gpu-gib", type=float, default=40.0)
    prepare_command.add_argument("--budget", type=int, choices=BUDGETS, default=300)
    prepare_command.add_argument("--note", default="")
    run = commands.add_parser("run")
    run.add_argument("--run", required=True)
    run.add_argument("--stage", required=True, choices=("search", "evaluate", "preflight"))
    run.add_argument("--condition", required=True)
    run.add_argument("--shard", default="0/1")
    run.add_argument("--device", default="cuda:0")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    C.setup_environment()
    if args.command == "prepare":
        output = prepare(args.parent_run, args.output, variant=args.variant,
                         min_free_gpu_gib=args.min_free_gpu_gib, budget=args.budget,
                         note=args.note)
        print(output, flush=True)
    else:
        run_stage(args.run, args.stage, args.condition, shard=args.shard, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

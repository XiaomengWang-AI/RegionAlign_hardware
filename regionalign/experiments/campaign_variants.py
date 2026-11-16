"""Separately frozen campaign variants and evaluator-side eligibility rules.

All variants retain the existing legality verdict and add constraints by
conjunction.  Search adapters select within those constraints; they never
replace the evaluator's verdict.  Equal exposure means the first 4096 stored
elements of each registered deployed page, independent of scores or gradients.
INT4 therefore exposes its predefined first half-page.  The random arm keeps
its original seeded full-page layout, then searches the exposed part of that
layout; it is not allowed to choose a different random placement.

``no_gradient`` supplies no measured gradient and records zero gradient calls;
``static_information`` retains initial information but disables later refresh;
``fixed_columns`` constrains only down_proj input-column permutations.
``distinct_weights`` adds at most one faulty bit per stored element.  The
existing search already selects at most one bit per element, so this last
variant is a legality control and need not change the selected artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from pathlib import Path
import shutil
from unittest.mock import patch

from regionalign.experiments import campaign_protocol as B
from regionalign.harness import core as P, region as R

MODULE_SOURCE = Path("regionalign/experiments/campaign_variants.py")
VARIANTS = ("no_gradient", "static_information", "fixed_columns", "equal_exposure",
            "distinct_weights")
PARAMETERS = {
    "no_gradient": {"measured_gradients": False},
    "static_information": {"dynamic_information_refresh": False},
    "fixed_columns": {"down_proj_input_column_permutations": False},
    "equal_exposure": {"exposed_stored_elements": 4096,
                       "exposure_selection": "first_elements_of_deployed_registered_page"},
    "distinct_weights": {"max_faulty_bits_per_stored_element": 1},
}
_stack = None
_plan_hash = None
_original_check = R.check_region_attack
_original_columns_movable = R.columns_movable


def _check_exposure_geometry(plan):
    for condition in plan["conditions"]:
        for region in condition["regions"]:
            rows, columns = R.tensor_shape(region["kind"], condition["dims"])
            first = region["page"] * R.elements_per_page(region["bits"], region["page_bytes"])
            if region["page_bytes"] != 4096 or min(rows * columns - first,
                    R.elements_per_page(region["bits"], region["page_bytes"])) < 4096:
                raise ValueError("equal exposure requires at least 4096 elements in every registered 4 KiB page")


def _application_arms(variant):
    return ["noperm", "local", "random"] if variant in ("equal_exposure", "distinct_weights") else ["local"]


def _application_kinds(variant):
    return ["down_proj"] if variant == "fixed_columns" else ["q_proj", "v_proj", "gate_proj", "down_proj"]


def _reference_baseline(run, parent_hash, scope, minimum):
    run, reference, digest = B._read_registered_plan(run)
    policy = reference.get("campaign_policy", {})
    if (reference["scope"] != scope or policy.get("variant") != "baseline"
            or policy.get("parent_plan_sha256") != parent_hash
            or policy.get("resource_policy", {}).get("min_free_gpu_gib") != minimum):
        raise ValueError("reference must be a baseline campaign with the same parent and resource policy")
    B._validate_policy(run, reference)
    return run, reference, digest


def _compare_reference(plan, reference, variant, *, selected):
    keys = ("models", "datasets", "repeat_seeds", "search_cost", "search_k", "evaluation_k",
            "clean_gate_nll", "mmlu_score_gate", "vqa_score_gate")
    if any(plan.get(key) != reference.get(key) for key in keys):
        raise ValueError("variant data, model, objective or evaluation identity differs from its baseline")
    expected = copy.deepcopy(reference["conditions"])
    if selected:
        for condition in expected:
            condition["regions"] = [r for r in condition["regions"]
                                    if r["kind"] in _application_kinds(variant)]
    if plan["conditions"] != expected:
        raise ValueError("variant conditions differ from the registered baseline application")


def _item_registration(plan):
    backend = B._backend(plan["scope"])
    registered = {}
    for condition in plan["conditions"]:
        rows = []
        for item in backend.planned_items(condition, plan):
            row = {key: item[key] for key in ("name", "repeat", "seed", "arm")}
            row["region_id"] = item["region"]["id"]
            if "bootstrap_seed" in item:
                row["bootstrap_seed"] = item["bootstrap_seed"]
            rows.append(row)
        if len({r["name"] for r in rows}) != len(rows):
            raise ValueError("variant registration has duplicate work items")
        registered[condition["id"]] = rows
    return registered


def _apply_application(plan, variant, reference_run, reference_hash):
    """Set actual driver inputs before the first plan/status publication."""
    plan["arms"] = _application_arms(variant)
    kinds = _application_kinds(variant)
    for condition in plan["conditions"]:
        condition["regions"] = [r for r in condition["regions"] if r["kind"] in kinds]
    count = 3 if variant == "fixed_columns" else 12
    if len(plan["repeat_seeds"]) != 3 or any(len(c["regions"]) != count for c in plan["conditions"]):
        raise ValueError("variant requires the frozen twelve-region, three-repeat parent design")
    plan["region_design"]["kinds"] = kinds
    plan["regions_per_condition"] = count
    plan["work_items_per_condition"] = count * 3 * len(plan["arms"])
    registered = _item_registration(plan)
    counts = {cid: len(rows) for cid, rows in registered.items()}
    if set(counts.values()) != {plan["work_items_per_condition"]}:
        raise ValueError("driver items disagree with variant work item counts")
    available = [c for c in plan["conditions"] if c.get("availability", "available") == "available"]
    plan["registered_available_searches"] = sum(counts[c["id"]] for c in available)
    application = {
        "arms": list(plan["arms"]), "region_kinds": kinds,
        "reference_baseline_run": str(reference_run),
        "reference_baseline_plan_sha256": reference_hash,
        "registered_items": registered,
        "registered_items_sha256": B.M.object_sha256(registered),
        "expected_work_items_by_condition": counts,
        "registered_available_searches": plan["registered_available_searches"],
    }
    plan["application"] = application
    plan["campaign_policy"]["application_sha256"] = B.M.object_sha256(application)


def _validate_application(plan):
    policy = plan["campaign_policy"]
    variant = policy["variant"]
    application = plan.get("application")
    if not isinstance(application, dict) or policy.get("application_sha256") != B.M.object_sha256(application):
        raise ValueError("variant application identity is missing or changed")
    _, reference, digest = _reference_baseline(application.get("reference_baseline_run"),
        policy["parent_plan_sha256"], plan["scope"], policy["resource_policy"]["min_free_gpu_gib"])
    if application.get("reference_baseline_plan_sha256") != digest:
        raise ValueError("registered baseline reference changed")
    _compare_reference(plan, reference, variant, selected=True)
    expected = copy.deepcopy(plan)
    _apply_application(expected, variant, application["reference_baseline_run"], digest)
    keys = ("arms", "region_design", "regions_per_condition", "work_items_per_condition",
            "registered_available_searches", "application")
    if any(plan.get(key) != expected.get(key) for key in keys):
        raise ValueError("variant application differs from actual registered driver work items")


def prepare(parent_run, output, *, variant, reference_baseline_run,
            min_free_gpu_gib=40.0, note=""):
    """Register an independent variant plan without rewriting any existing plan."""
    if variant not in VARIANTS:
        raise ValueError("unsupported campaign variant")
    _, inherited, parent_hash = B._read_registered_plan(parent_run)
    reference_run, reference, reference_hash = _reference_baseline(reference_baseline_run,
        parent_hash, inherited["scope"], B._memory_threshold(min_free_gpu_gib))
    _compare_reference(inherited, reference, variant, selected=False)
    if variant == "equal_exposure":
        _check_exposure_geometry(inherited)
    backend = B._backend(inherited["scope"])
    freeze_sources = backend._copy_sources
    write_once = B.M.write_json_once

    def sources_with_variant(run):
        records = freeze_sources(run)
        live = B.C.PROJECT / MODULE_SOURCE
        saved = Path(run) / "source" / str(MODULE_SOURCE).replace("/", "__")
        shutil.copyfile(live, saved)
        digest = B.M.sha256(live)
        if B.M.sha256(saved) != digest:
            raise RuntimeError("variant source changed during freezing")
        records.append({"path": str(MODULE_SOURCE), "snapshot": str(saved.relative_to(run)),
                        "sha256": digest, "bytes": live.stat().st_size})
        return records

    application_identity = {}
    def register_parameters(path, value):
        if Path(path).name == "plan.json" and isinstance(value, dict):
            policy = value.get("campaign_policy")
            if policy and policy.get("variant") == variant:
                value["campaign_policy"]["parameters"].update(PARAMETERS[variant])
                value["campaign_policy"]["variant_source"] = str(MODULE_SOURCE)
                _apply_application(value, variant, reference_run, reference_hash)
                application_identity.update({
                    "application_sha256": value["campaign_policy"]["application_sha256"],
                    "reference_baseline_plan_sha256": reference_hash,
                })
        elif Path(path).name == "status.json" and isinstance(value, dict):
            value.update(application_identity)
        return write_once(path, value)

    with (patch.object(B, "VARIANTS", (*B.VARIANTS, *VARIANTS)),
          patch.object(backend, "_copy_sources", sources_with_variant),
          patch.object(B.M, "write_json_once", register_parameters)):
        return B.prepare(parent_run, output, variant=variant,
                         min_free_gpu_gib=min_free_gpu_gib, note=note)


def eligibility_reasons(region, flips, policy, *, spec=None, dims=None):
    """Additional evaluator rules; base geometry/value/bit rules remain mandatory."""
    variant = policy["variant"]
    reasons = []
    normalized = P.normalize_flips(flips)
    if variant == "distinct_weights":
        elements = [(name, element) for name, element, _ in normalized]
        if len(set(elements)) != len(elements):
            reasons.append("campaign permits at most one faulty bit per stored element")
    if variant == "equal_exposure":
        first = region["page"] * R.elements_per_page(region["bits"], region["page_bytes"])
        for name, element, _ in normalized:
            if name == region["tensor"] and not first <= element < first + 4096:
                reasons.append("bit is outside the predefined 4096-element exposure window")
                break
    if variant == "fixed_columns" and region["kind"] == "down_proj" and spec is not None:
        spec = P.validate_spec(spec, dims)
        order = P.axis_orders(spec, dims).get((region["tensor"], 1))
        if order is not None:
            reasons.append("campaign fixes down_proj input-column coordinates")
    return reasons


def check_region_attack(rt, region, flips, k, *, policy):
    """Conjoin extension eligibility with the shared evaluator's original verdict."""
    verdict = _original_check(rt, region, flips, k)
    try:
        verdict["reasons"].extend(eligibility_reasons(region, flips, policy))
    except (ValueError, KeyError, TypeError) as error:
        verdict["reasons"].append(str(error))
    verdict["valid"] = bool(verdict["valid"] and not verdict["reasons"])
    return verdict


def exposed_region(region):
    """Geometry of the first exposed 4096 stored elements; retain the public ID."""
    if region["bits"] != 4:
        return dict(region)
    return {**region, "page": region["page"] * 2, "page_bytes": 2048}


def context_class(base, policy):
    class CampaignContext(base):
        def __init__(self, rt, region, arm, *args, **kwargs):
            self._registered_region = dict(region)
            effective = exposed_region(region) if policy["variant"] == "equal_exposure" else region
            # The random control's layout is still the full-page seeded layout.
            super().__init__(rt, region if arm == "random" else effective, arm, *args, **kwargs)
            if policy["variant"] == "equal_exposure":
                self.segments = R.page_segments(effective, self.d)
                if arm != "random":
                    self.primary = R.primary_segment(effective, self.d)

        def grad(self, flips=()):
            if policy["variant"] == "no_gradient":
                import torch
                self._flips(flips)
                return torch.zeros_like(self.weight(), dtype=torch.float32)
            return super().grad(flips)

        def check(self, spec, attacks):
            verdicts = super().check(spec, attacks)
            extra = eligibility_reasons(self._registered_region, [], policy, spec=spec, dims=self.d)
            for info in verdicts.values():
                info["reasons"].extend(extra)
                info["valid"] = bool(info["valid"] and not info["reasons"])
            return verdicts
    return CampaignContext


def adapt_search(module, policy):
    """Adapt information availability in the frozen shared search implementation."""
    if policy["variant"] == "static_information":
        def no_refresh(ctx, weights, state, group, limit):
            group.refreshed = True
        module.refresh = no_refresh
    return module


def configure_from_plan(run):
    """Configure one independently executing variant process before formal loading."""
    global _stack, _plan_hash
    run, plan, digest = B._read_registered_plan(run)
    policy = plan.get("campaign_policy", {})
    variant = policy.get("variant")
    if variant not in VARIANTS:
        raise ValueError("not a registered campaign variant plan")
    if policy.get("parameters") != {"search_budget": 300, **PARAMETERS[variant]}:
        raise ValueError("variant parameters differ from the registered implementation")
    _validate_application(plan)
    records = [r for r in plan["sources"] if r["path"] == str(MODULE_SOURCE)]
    if (policy.get("variant_source") != str(MODULE_SOURCE) or len(records) != 1
            or B.M.sha256(__file__) != records[0]["sha256"]):
        raise ValueError("executing variant source differs from the frozen plan")
    if _plan_hash is not None:
        if _plan_hash != digest:
            raise RuntimeError("a different variant plan requires an independent process")
        return B.configure_from_plan(run)
    if variant == "equal_exposure":
        _check_exposure_geometry(plan)
    stack = ExitStack()
    try:
        stack.enter_context(patch.object(B, "VARIANTS", (*B.VARIANTS, *VARIANTS)))
        loaded = B.configure_from_plan(run)
        backend = B._backend(plan["scope"])
        original_loader = backend._search_module
        def search_loader(path):
            return adapt_search(original_loader(path), policy)
        stack.enter_context(patch.object(backend, "_search_module", search_loader))
        base_context = R.RegionContext
        campaign_context = context_class(base_context, policy)
        stack.enter_context(patch.object(R, "RegionContext", campaign_context))
        if plan["scope"] == "vqa_regionalign_dev":
            stack.enter_context(patch.object(backend.V, "VQARegionContext", campaign_context))
        def check(rt, region, flips, k):
            return check_region_attack(rt, region, flips, k, policy=policy)
        stack.enter_context(patch.object(R, "check_region_attack", check))
        if variant == "fixed_columns":
            stack.enter_context(patch.object(R, "columns_movable", lambda region:
                False if region["kind"] == "down_proj" else _original_columns_movable(region)))
        original_artifact_reasons = backend._artifact_reasons
        def artifact_reasons(rt, artifact, condition, item, registered_plan):
            reasons = original_artifact_reasons(rt, artifact, condition, item, registered_plan)
            try:
                reasons.extend(eligibility_reasons(item["region"], [], policy,
                    spec=artifact.get("permutation"), dims=condition["dims"]))
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(str(error))
            return reasons
        stack.enter_context(patch.object(backend, "_artifact_reasons", artifact_reasons))
        _stack, _plan_hash = stack, digest
        return loaded
    except BaseException:
        stack.close()
        B._reset_configuration()
        raise


def _reset_configuration():
    global _stack, _plan_hash
    if _stack is not None:
        _stack.close()
    _stack = _plan_hash = None
    B._reset_configuration()


def run_stage(run, stage, condition, *, shard="0/1", device="cuda:0"):
    configure_from_plan(run)
    return B.run_stage(run, stage, condition, shard=shard, device=device)


def main(argv=None):
    root = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--parent-run", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--variant", required=True, choices=VARIANTS)
    create.add_argument("--reference-baseline-run", required=True)
    create.add_argument("--min-free-gpu-gib", type=float, default=40.0)
    create.add_argument("--note", default="")
    run = commands.add_parser("run")
    run.add_argument("--run", required=True)
    run.add_argument("--stage", required=True, choices=("search", "evaluate", "preflight"))
    run.add_argument("--condition", required=True)
    run.add_argument("--shard", default="0/1")
    run.add_argument("--device", default="cuda:0")
    args = root.parse_args(argv)
    B.C.setup_environment()
    if args.command == "prepare":
        print(prepare(args.parent_run, args.output, variant=args.variant,
                      reference_baseline_run=args.reference_baseline_run,
                      min_free_gpu_gib=args.min_free_gpu_gib, note=args.note), flush=True)
    else:
        run_stage(args.run, args.stage, args.condition, shard=args.shard, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

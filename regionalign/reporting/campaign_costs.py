"""Read-only registered campaign costs, development damage and matched controls.

Only registry/plan/status JSON and registered artifact/eval JSON are read. This
module imports no model/runtime code, opens no panels and writes no access logs.
Every invocation publishes CSV/JSON in a fresh directory. ``progress`` accepts
pending work; ``complete`` returns nonzero until all available registered files
are present and identity bound. Scientific failures remain explicit and do not
become missing coverage. Unavailable conditions remain outside executed scope.

Costs are recorded search-panel requests, not GPU kernels or end-to-end costs:
``calls.eval`` counts forward-only panel requests (including activation profiles)
and ``calls.grad`` counts forward/backward panel requests. Consequently the
reported forward/backward panel passes are eval+grad and grad; forward-equivalent
cost is recomputed using the frozen plan's coefficients and checked against
``spent`` and its budget. Loading, qualification and independent evaluation are
excluded. Missing or malformed counters remain null, never zero-filled.

Damage is taken only from valid independent dev results, relative to their own
rearranged unfaulted layout. WikiText uses perplexity ratio; MMLU and VQA use
relative score drop. Pairing keeps model, format, task, region, repeat, bit
budget, checkpoint, frozen panels and bootstrap identity together. Local-only
variants may use only their explicitly registered reference baseline's noperm.
No pooled ranking across tasks/models or best-of-budget selection is computed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


DEFAULT_BASE = Path(__file__).resolve().parents[1] / "results/campaigns/20261110T204835_083076Z_all_experiments"
REGISTRIES = ("baseline_plans.json", "budget_plans.json", "variant_plans.json")
SCOPES = {"llm_regionalign_dev", "vqa_regionalign_dev"}
TASKS = {"mmlu", "wikitext", "okvqa", "textvqa"}
SCOPE = "regionalign_campaign_cost_coverage_v1"


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_record(path):
    serialized = Path(path).read_bytes()
    return json.loads(serialized), hashlib.sha256(serialized).hexdigest()


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def registration(condition, plan):
    rows = []
    for region in condition["regions"]:
        for repeat, seed in enumerate(plan["repeat_seeds"]):
            for arm in plan["arms"]:
                row = {"name": f"{region['id']}__r{repeat}__{arm}.json", "region": region,
                       "region_id": region["id"], "repeat": repeat, "seed": seed, "arm": arm}
                if plan["scope"] == "llm_regionalign_dev":
                    row["bootstrap_seed"] = condition["bootstrap_seeds"][f"{region['id']}__r{repeat}"]
                rows.append(row)
    if len({row["name"] for row in rows}) != len(rows) or len(rows) != plan["work_items_per_condition"]:
        raise ValueError("plan work item registration is inconsistent")
    application = plan.get("application")
    if application:
        declared = [{key: value for key, value in row.items() if key != "region"} for row in rows]
        if application["registered_items"].get(condition["id"]) != declared:
            raise ValueError("variant application differs from its registered driver items")
        if (application.get("registered_items_sha256") != object_hash(application["registered_items"])
                or plan["campaign_policy"].get("application_sha256") != object_hash(application)):
            raise ValueError("variant application hashes differ from its plan")
    return rows


def load_runs(base):
    runs, registry_records = {}, []
    for name in REGISTRIES:
        path = base / name
        registry, digest = read_record(path)
        if not isinstance(registry, dict):
            raise ValueError(f"registry must be an object: {path}")
        registry_records.append({"path": str(path), "sha256": digest})
        for alias, raw in registry.items():
            run = Path(raw)
            run = (run if run.is_absolute() else base / run).resolve()
            if run in runs:
                runs[run]["registry_aliases"].append(f"{name}:{alias}")
                continue
            plan, plan_hash = read_record(run / "plan.json")
            status, _ = read_record(run / "status.json")
            policy = plan.get("campaign_policy") or {}
            if (plan.get("scope") not in SCOPES or plan.get("schema_version") != 2
                    or plan.get("development_only") is not True or plan.get("splits") != ["search", "dev"]
                    or status.get("plan_sha256") != plan_hash
                    or policy.get("scope") != "regionalign_campaign_dev"
                    or plan.get("search_budget") not in (50, 100, 200, 300)
                    or policy.get("parameters", {}).get("search_budget") != plan.get("search_budget")):
                raise ValueError(f"not a registered campaign development plan: {run}")
            coefficients = plan.get("search_cost")
            if (not isinstance(coefficients, dict) or set(coefficients) != {"eval", "grad"}
                    or any(type(v) is not int or v < 1 for v in coefficients.values())):
                raise ValueError("invalid frozen cost coefficients")
            if len({c["id"] for c in plan["conditions"]}) != len(plan["conditions"]):
                raise ValueError("duplicate registered condition")
            runs[run] = {"run": run, "plan": plan, "plan_sha256": plan_hash,
                         "registry_aliases": [f"{name}:{alias}"]}
    return list(runs.values()), registry_records


def inspect_file(path, expected):
    if not path.exists():
        return None, None, "pending", []
    try:
        value, digest = read_record(path)
        if not isinstance(value, dict):
            raise ValueError("evidence is not an object")
        errors = [f"identity mismatch: {key}" for key, wanted in expected.items() if value.get(key) != wanted]
        return value, digest, "identity_error" if errors else "present", errors
    except (OSError, ValueError, TypeError) as error:
        return None, None, "read_error", [str(error)]


def accounting(artifact, plan):
    empty = {"calls_eval": None, "calls_grad": None, "forward_panel_passes": None,
             "backward_panel_passes": None, "reported_spent": None, "recomputed_cost": None,
             "accounting_location": None, "accounting_valid": None, "within_budget": None,
             "accounting_errors": []}
    if artifact is None:
        return empty
    nested = artifact.get("search") if isinstance(artifact.get("search"), dict) else {}
    source = artifact if "calls" in artifact or "spent" in artifact else nested
    location = "root" if source is artifact else "search"
    calls, spent = source.get("calls"), source.get("spent")
    if (not isinstance(calls, dict) or set(calls) != {"eval", "grad"}
            or any(type(v) is not int or v < 0 for v in calls.values())):
        return {**empty, "accounting_valid": False, "accounting_errors": ["missing or invalid recorded calls"]}
    errors = []
    if source is artifact and "calls" in nested and nested["calls"] != calls:
        errors.append("conflicting root and nested call counts")
    if source is artifact and "spent" in nested and nested["spent"] != spent:
        errors.append("conflicting root and nested spent")
    cost = sum(plan["search_cost"][key] * calls[key] for key in calls)
    if type(spent) is not int or spent != cost:
        errors.append("reported spent differs from recorded calls")
    within = cost <= plan["search_budget"]
    if not within:
        errors.append("recorded cost exceeds frozen budget")
    return {"calls_eval": calls["eval"], "calls_grad": calls["grad"],
            "forward_panel_passes": calls["eval"] + calls["grad"], "backward_panel_passes": calls["grad"],
            "reported_spent": number(spent), "recomputed_cost": cost, "accounting_location": location,
            "accounting_valid": not errors, "within_budget": within, "accounting_errors": errors}


def sample_hash(artifact, condition, item, scope):
    if not artifact:
        return None
    if scope == "llm_regionalign_dev":
        sample = artifact.get("bootstrap") or {}
        sids = sample.get("sampled_sids")
        if (sample.get("seed") != item["bootstrap_seed"] or not isinstance(sids, list)
                or sample.get("sampled_sids_sha256") != object_hash(sids)):
            return None
        return sample["sampled_sids_sha256"]
    expected = [row for row in condition.get("search_bootstraps", []) if row.get("repeat") == item["repeat"]]
    sample = artifact.get("search_sample")
    return sample.get("sid_sha256") if len(expected) == 1 and sample == expected[0] else None


def rows_for_run(record):
    run, plan, digest = record["run"], record["plan"], record["plan_sha256"]
    policy = plan["campaign_policy"]
    items, damage = [], []
    for condition in plan["conditions"]:
        if condition["task"] not in TASKS:
            raise ValueError("unsupported task metric")
        available = condition.get("availability", "available") == "available"
        for item in registration(condition, plan):
            base = {"run": str(run), "plan_sha256": digest, "variant": policy["variant"],
                    "budget": plan["search_budget"], "model": condition["model"], "format": condition["format"],
                    "task": condition["task"], "condition": condition["id"], "availability": "available" if available else "unavailable",
                    "region_id": item["region_id"], "region_kind": item["region"].get("kind"),
                    "region_sha256": object_hash(item["region"]),
                    "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"], "item_name": item["name"]}
            expected = {"schema_version": 2, "scope": plan["scope"], "plan_sha256": digest, "condition": condition["id"],
                        "region": item["region"], "repeat": item["repeat"], "seed": item["seed"], "arm": item["arm"],
                        "campaign_scope": policy["scope"], "campaign_variant": policy["variant"],
                        "campaign_policy_sha256": object_hash(policy)}
            artifact_path, eval_path = run / "artifacts" / condition["id"] / item["name"], run / "eval" / condition["id"] / item["name"]
            artifact_expected = {**expected, **({"bootstrap_seed": item["bootstrap_seed"]} if "bootstrap_seed" in item else {})}
            artifact, artifact_hash, search_state, search_errors = inspect_file(artifact_path, artifact_expected)
            evaluation, eval_hash, eval_state, eval_errors = inspect_file(eval_path, expected)
            if not available:
                search_state = eval_state = "unavailable"
            if available and evaluation is not None and (search_state != "present" or artifact_hash is None
                    or evaluation.get("artifact_file_sha256") != artifact_hash):
                eval_state = "identity_error"
                eval_errors.append("evaluation does not bind the current source artifact")
            cost = accounting(artifact if search_state == "present" else None, plan)
            row = {**base, "artifact_path": str(artifact_path), "artifact_sha256": artifact_hash,
                   "eval_path": str(eval_path), "eval_sha256": eval_hash, "search_state": search_state,
                   "eval_state": eval_state, "search_errors": search_errors, "eval_errors": eval_errors,
                   "search_failed": "error" in artifact if artifact is not None else None,
                   "evaluation_status": evaluation.get("status") if evaluation else None,
                   "sample_sha256": sample_hash(artifact, condition, item, plan["scope"]) if search_state == "present" else None,
                   "checkpoint_sha256": condition.get("checkpoint_manifest_sha256"),
                   "panels_sha256": object_hash(condition.get("panels")), **cost}
            items.append(row)
            for k in plan["evaluation_k"]:
                metric = "ppl_ratio" if condition["task"] == "wikitext" else "relative_score_drop"
                result = ((evaluation or {}).get("results") or {}).get(str(k))
                status = "unavailable" if not available else "pending" if eval_state == "pending" else eval_state
                valid, success, primary, delta = None, None, None, None
                if available and eval_state == "present":
                    report_status = evaluation.get("status")
                    if report_status != "complete":
                        status = report_status or "evaluation_error"
                    elif not isinstance(result, dict) or type(result.get("valid")) is not bool:
                        status = "missing_budget_result"
                    elif not result["valid"]:
                        status, valid = "invalid", False
                    elif ((evaluation.get("rearranged") or {}).get("gate") is not True
                          or result.get("damage_baseline") not in ("same_rearranged_unfaulted_layout", "rearranged_unfaulted")):
                        status, valid = "inconsistent_damage_baseline", False
                    else:
                        status, valid = "valid", True
                        primary, delta = number(result.get(metric)), number(result.get("delta_nll"))
                        success = result.get("success") if type(result.get("success")) is bool else None
                damage.append({**base, "k": k, "damage_metric": metric, "state": status, "valid": valid,
                               "damage": primary, "delta_nll": delta, "success": success,
                               "sample_sha256": row["sample_sha256"], "recomputed_cost": row["recomputed_cost"],
                               "accounting_valid": row["accounting_valid"], "checkpoint_sha256": row["checkpoint_sha256"],
                               "panels_sha256": row["panels_sha256"]})
    return items, damage


def paired_rows(damage, records):
    by_key = {(row["run"], row["model"], row["format"], row["task"], row["region_id"], row["repeat"], row["k"], row["arm"]): row for row in damage}
    records_by_run = {str(record["run"]): record for record in records}
    pairs = []
    for local in damage:
        if local["arm"] != "local":
            continue
        source = records_by_run[local["run"]]
        control_run, origin = local["run"], "same_plan"
        if "noperm" not in source["plan"]["arms"]:
            application = source["plan"].get("application") or {}
            raw = application.get("reference_baseline_run")
            control_run = str(Path(raw).resolve()) if raw else None
            reference = records_by_run.get(control_run)
            if (not reference or reference["plan_sha256"] != application.get("reference_baseline_plan_sha256")
                    or reference["plan"]["campaign_policy"]["variant"] != "baseline"
                    or reference["plan"]["search_budget"] != local["budget"]
                    or reference["plan"]["search_cost"] != source["plan"]["search_cost"]):
                control_run = None
            origin = "registered_reference_baseline"
        key = (control_run, local["model"], local["format"], local["task"], local["region_id"], local["repeat"], local["k"], "noperm")
        control = by_key.get(key)
        status, difference = "control_not_registered", None
        if control:
            if local["availability"] == "unavailable" or control["availability"] == "unavailable":
                status = "unavailable"
            elif local["state"] == "pending" or control["state"] == "pending":
                status = "pending"
            elif local["accounting_valid"] is not True or control["accounting_valid"] is not True:
                status = "invalid_accounting"
            elif (local["sample_sha256"] is None or local["sample_sha256"] != control["sample_sha256"]
                  or local["checkpoint_sha256"] is None or local["checkpoint_sha256"] != control["checkpoint_sha256"]
                  or local["panels_sha256"] != control["panels_sha256"] or local["seed"] != control["seed"]
                  or local["region_sha256"] != control["region_sha256"]):
                status = "unmatched_inputs"
            elif local["valid"] is not True or control["valid"] is not True:
                status = "invalid_evaluation"
            elif local["damage"] is None or control["damage"] is None:
                status = "unavailable_metric"
            else:
                status, difference = "matched", local["damage"] - control["damage"]
        pairs.append({key: local[key] for key in ("run", "plan_sha256", "variant", "budget", "model", "format", "task", "region_id", "region_sha256", "repeat", "k", "damage_metric")} | {
            "state": status, "control_origin": origin, "control_run": control_run,
            "control_plan_sha256": control.get("plan_sha256") if control else None,
            "local_damage": local["damage"], "noperm_damage": control.get("damage") if control else None,
            "local_minus_noperm": difference, "local_cost": local["recomputed_cost"],
            "noperm_cost": control.get("recomputed_cost") if control else None})
    return pairs


def coverage_rows(items):
    groups = defaultdict(list)
    for row in items:
        groups[tuple(row[key] for key in ("run", "plan_sha256", "variant", "budget", "model", "format", "task", "arm", "availability"))].append(row)
    result = []
    for key, rows in groups.items():
        search_counts, eval_counts = Counter(row["search_state"] for row in rows), Counter(row["eval_state"] for row in rows)
        costs = [row["recomputed_cost"] for row in rows if row["recomputed_cost"] is not None]
        available = rows[0]["availability"] == "available"
        complete = available and search_counts["present"] == len(rows) and eval_counts["present"] == len(rows)
        result.append(dict(zip(("run", "plan_sha256", "variant", "budget", "model", "format", "task", "arm", "availability"), key)) | {
            "coverage_state": "complete" if complete else "pending" if available else "unavailable",
            "registered_items": len(rows), "search_present": search_counts["present"], "eval_present": eval_counts["present"],
            "search_pending": search_counts["pending"], "eval_pending": eval_counts["pending"],
            "search_evidence_errors": sum(n for state, n in search_counts.items() if state not in ("pending", "present", "unavailable")),
            "eval_evidence_errors": sum(n for state, n in eval_counts.items() if state not in ("pending", "present", "unavailable")),
            "search_failed_items": sum(row["search_failed"] is True for row in rows),
            "evaluation_status_counts": dict(Counter(row["evaluation_status"] or "pending" for row in rows)),
            "recorded_cost_items": len(costs), "valid_accounting_items": sum(row["accounting_valid"] is True for row in rows),
            "observed_cost_sum": sum(costs) if costs else None, "mean_observed_cost": sum(costs) / len(costs) if costs else None})
    return result


def collect(base):
    base = Path(base).resolve()
    records, registries = load_runs(base)
    items, damage = [], []
    for record in records:
        current_items, current_damage = rows_for_run(record)
        items.extend(current_items)
        damage.extend(current_damage)
    coverage = coverage_rows(items)
    pairs = paired_rows(damage, records)
    available = [row for row in items if row["availability"] == "available"]
    complete = bool(available) and all(row["search_state"] == "present" and row["eval_state"] == "present" for row in available)
    summary = {"schema_version": 1, "scope": SCOPE, "created_at": datetime.now(timezone.utc).isoformat(),
        "base": str(base), "coverage_complete_available_scope": complete,
        "matrix_complete": complete and all(row["availability"] == "available" for row in items),
        "registries": registries, "plan_count": len(records), "registered_items": len(items),
        "available_items": len(available), "unavailable_items": len(items) - len(available),
        "pending_available_items": sum(row["search_state"] == "pending" or row["eval_state"] == "pending" for row in available),
        "evidence_error_items": sum(row["search_state"] not in ("present", "pending") or row["eval_state"] not in ("present", "pending") for row in available),
        "cost_accounting_error_items": sum(row["accounting_valid"] is False for row in available),
        "damage_state_counts": dict(Counter(row["state"] for row in damage)),
        "pair_state_counts": dict(Counter(row["state"] for row in pairs)),
        "plans": [{"run": str(r["run"]), "plan_sha256": r["plan_sha256"], "registry_aliases": r["registry_aliases"]} for r in records],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cost_boundary": "recorded search-panel requests only; excludes loading, qualification and independent evaluation",
        "damage_boundary": "independent dev evaluation against the same rearranged unfaulted layout; no test data or pooled model/task ranking"}
    return {"summary": summary, "items": items, "damage_by_k": damage, "matched_local_noperm": pairs, "coverage": coverage}


def write_report(base, output, mode="progress"):
    if mode not in ("progress", "complete"):
        raise ValueError("mode must be progress or complete")
    report = collect(base)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key in ("items", "damage_by_k", "matched_local_noperm", "coverage"):
        rows = report[key]
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with (output / f"{key}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    report["summary"]["mode"] = mode
    with (output / "summary.json").open("x") as handle:
        json.dump(report["summary"], handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("progress", "complete"), default="progress")
    args = parser.parse_args(argv)
    summary = write_report(args.base, args.output, args.mode)
    print(json.dumps({key: summary[key] for key in ("coverage_complete_available_scope", "matrix_complete", "pending_available_items", "evidence_error_items")}, sort_keys=True))
    return int(args.mode == "complete" and not summary["coverage_complete_available_scope"])


if __name__ == "__main__":
    raise SystemExit(main())

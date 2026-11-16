"""Cost arithmetic, incomplete coverage and matched-control reporting without models."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from regionalign.reporting import campaign_costs as S


class CampaignCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.registries = {name: {} for name in S.REGISTRIES}
        self.save_registries()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def save_registries(self):
        for name, value in self.registries.items():
            self.write(self.base / name, value)

    def add_plan(self, name="baseline", budget=300, variant="baseline", task="mmlu", model="tiny",
                 arms=("noperm", "local"), unavailable=False, reference=None):
        run = self.base / name
        region = {"id": "r0", "kind": "q_proj", "tensor": "model.layers.0.self_attn.q_proj.weight"}
        condition = {"id": f"{model}__int8__{task}", "model": model, "format": "int8", "task": task,
            "availability": "unavailable" if unavailable else "available", "regions": [region],
            "bootstrap_seeds": {"r0__r0": 17}, "checkpoint_manifest_sha256": "c" * 64,
            "panels": {"search": {"sha256": "a" * 64}, "dev": {"sha256": "b" * 64}}}
        vqa = task in ("okvqa", "textvqa")
        if vqa:
            condition["search_bootstraps"] = [{"repeat": 0, "seed": 17, "sid_sha256": "d" * 64}]
        policy = {"scope": "regionalign_campaign_dev", "variant": variant,
                  "parameters": {"search_budget": budget}}
        plan = {"schema_version": 2, "scope": "vqa_regionalign_dev" if vqa else "llm_regionalign_dev", "development_only": True,
            "splits": ["search", "dev"], "campaign_policy": policy, "conditions": [condition],
            "arms": list(arms), "repeat_seeds": [7], "work_items_per_condition": len(arms),
            "search_budget": budget, "search_cost": {"eval": 1, "grad": 3}, "evaluation_k": [1, 2]}
        if reference:
            reference_run, _reference = reference
            declared = [{"name": f"r0__r0__{arm}.json", "region_id": "r0", "repeat": 0,
                         "seed": 7, "arm": arm, **({} if vqa else {"bootstrap_seed": 17})} for arm in arms]
            application = {"registered_items": {condition["id"]: declared},
                "reference_baseline_run": str(reference_run),
                "reference_baseline_plan_sha256": S.read_record(reference_run / "plan.json")[1]}
            application["registered_items_sha256"] = S.object_hash(application["registered_items"])
            plan["application"] = application
            policy["application_sha256"] = S.object_hash(application)
        self.write(run / "plan.json", plan)
        self.write(run / "status.json", {"plan_sha256": S.read_record(run / "plan.json")[1]})
        registry = "baseline_plans.json" if variant == "baseline" else "budget_plans.json" if variant == "budget" else "variant_plans.json"
        self.registries[registry][name] = str(run)
        self.save_registries()
        return run, plan

    def evidence(self, registered, arm, *, damage=None, valid=True, calls=None, spent=None,
                 status="complete", source_error=False, nested=False, sampled_sids=None):
        run, plan = registered
        condition, policy = plan["conditions"][0], plan["campaign_policy"]
        item = next(item for item in S.registration(condition, plan) if item["arm"] == arm)
        identity = {"schema_version": 2, "scope": plan["scope"],
            "plan_sha256": S.read_record(run / "plan.json")[1], "condition": condition["id"],
            "region": item["region"], "repeat": 0, "seed": 7, "arm": arm,
            "campaign_scope": policy["scope"], "campaign_variant": policy["variant"],
            "campaign_policy_sha256": S.object_hash(policy)}
        sids = sampled_sids or ["frozen-sample-0", "frozen-sample-1"]
        artifact = {**identity, "bootstrap_seed": 17,
                    "bootstrap": {"seed": 17, "sampled_sids": sids,
                                  "sampled_sids_sha256": S.object_hash(sids)}}
        if plan["scope"] == "vqa_regionalign_dev":
            artifact = {**identity, "search_sample": condition["search_bootstraps"][0]}
        if source_error:
            artifact["error"] = "retained search failure"
        else:
            counts = calls if calls is not None else {"eval": 10, "grad": 2}
            accounting = {"calls": counts, "spent": spent if spent is not None else counts["eval"] + 3 * counts["grad"]}
            if nested:
                artifact["search"] = accounting
            else:
                artifact.update(accounting)
        artifact_path = run / "artifacts" / condition["id"] / item["name"]
        self.write(artifact_path, artifact)
        primary = damage if damage is not None else .8 if arm == "local" else .2
        baseline = "rearranged_unfaulted" if plan["scope"] == "vqa_regionalign_dev" else "same_rearranged_unfaulted_layout"
        results = {str(k): {"valid": valid, "damage_baseline": baseline,
                   "relative_score_drop": primary, "ppl_ratio": primary, "delta_nll": primary / 2,
                   "success": primary >= .5} for k in plan["evaluation_k"]}
        report = {**identity, "status": status, "artifact_file_sha256": S.read_record(artifact_path)[1],
                  "rearranged": {"gate": valid}, "results": results if status == "complete" else {}}
        if status == "artifact_error":
            report["artifact_error"] = "retained rejection"
        self.write(run / "eval" / condition["id"] / item["name"], report)
        return artifact_path, run / "eval" / condition["id"] / item["name"]

    def test_pending_is_null_not_a_zero_cost_or_zero_damage(self):
        self.add_plan(budget=50, variant="budget")
        report = S.collect(self.base)
        self.assertFalse(report["summary"]["coverage_complete_available_scope"])
        self.assertEqual(report["summary"]["pending_available_items"], 2)
        self.assertTrue(all(row["recomputed_cost"] is None and row["calls_eval"] is None for row in report["items"]))
        self.assertTrue(all(row["damage"] is None and row["valid"] is None and row["state"] == "pending" for row in report["damage_by_k"]))
        self.assertTrue(all(row["state"] == "pending" and row["local_minus_noperm"] is None for row in report["matched_local_noperm"]))
        self.assertTrue(all(row["observed_cost_sum"] is None for row in report["coverage"]))

    def test_real_recorded_counts_and_matched_task_damage_are_recomputed(self):
        plan = self.add_plan()
        for arm in ("noperm", "local"):
            self.evidence(plan, arm)
        report = S.collect(self.base)
        self.assertTrue(report["summary"]["coverage_complete_available_scope"])
        for row in report["items"]:
            self.assertEqual((row["forward_panel_passes"], row["backward_panel_passes"], row["recomputed_cost"]), (12, 2, 16))
            self.assertTrue(row["accounting_valid"])
        for row in report["matched_local_noperm"]:
            self.assertEqual(row["state"], "matched")
            self.assertAlmostEqual(row["local_minus_noperm"], .6)
            self.assertEqual(row["damage_metric"], "relative_score_drop")

    def test_nested_counters_work_but_bad_spent_and_over_budget_remain_explicit(self):
        plan = self.add_plan(budget=50, variant="budget")
        self.evidence(plan, "local", nested=True)
        self.evidence(plan, "noperm", calls={"eval": 50, "grad": 1}, spent=52)
        report = S.collect(self.base)
        local = next(r for r in report["items"] if r["arm"] == "local")
        bad = next(r for r in report["items"] if r["arm"] == "noperm")
        self.assertEqual(local["accounting_location"], "search")
        self.assertEqual(local["recomputed_cost"], 16)
        self.assertEqual(bad["recomputed_cost"], 53)
        self.assertFalse(bad["within_budget"])
        self.assertFalse(bad["accounting_valid"])
        self.assertEqual(len(bad["accounting_errors"]), 2)
        self.assertTrue(report["summary"]["coverage_complete_available_scope"])
        self.assertTrue(all(p["state"] == "invalid_accounting" and p["local_minus_noperm"] is None for p in report["matched_local_noperm"]))

    def test_conflicting_nested_and_root_counters_are_not_silently_selected(self):
        value = {"calls": {"eval": 1, "grad": 0}, "spent": 1,
                 "search": {"calls": {"eval": 3, "grad": 0}, "spent": 3}}
        result = S.accounting(value, {"search_budget": 50, "search_cost": {"eval": 1, "grad": 3}})
        self.assertFalse(result["accounting_valid"])
        self.assertEqual(len(result["accounting_errors"]), 2)
        zero = S.accounting({"calls": {"eval": 0, "grad": 0}, "spent": 0},
                            {"search_budget": 50, "search_cost": {"eval": 1, "grad": 3}})
        self.assertEqual(zero["recomputed_cost"], 0)
        self.assertTrue(zero["accounting_valid"])

    def test_scientific_failure_is_complete_coverage_without_fabricated_damage(self):
        plan = self.add_plan()
        self.evidence(plan, "local", source_error=True, status="artifact_error")
        self.evidence(plan, "noperm", valid=False)
        report = S.collect(self.base)
        self.assertTrue(report["summary"]["coverage_complete_available_scope"])
        self.assertEqual(report["summary"]["cost_accounting_error_items"], 1)
        self.assertTrue(all(row["damage"] is None for row in report["damage_by_k"]))
        self.assertIn("artifact_error", report["summary"]["damage_state_counts"])
        self.assertIn("invalid", report["summary"]["damage_state_counts"])

    def test_bad_source_identity_invalidates_bound_report_damage(self):
        plan = self.add_plan()
        artifact, evaluation = self.evidence(plan, "local")
        source, _ = S.read_record(artifact)
        source["region"] = {"id": "wrong"}
        self.write(artifact, source)
        report, _ = S.read_record(evaluation)
        report["artifact_file_sha256"] = S.read_record(artifact)[1]
        self.write(evaluation, report)
        result = S.collect(self.base)
        local = next(r for r in result["items"] if r["arm"] == "local")
        self.assertEqual((local["search_state"], local["eval_state"]), ("identity_error", "identity_error"))
        self.assertTrue(all(r["valid"] is None for r in result["damage_by_k"] if r["arm"] == "local"))

    def test_stale_artifact_hash_prevents_valid_damage(self):
        plan = self.add_plan()
        artifact, _ = self.evidence(plan, "local")
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        result = S.collect(self.base)
        self.assertEqual(next(r for r in result["items"] if r["arm"] == "local")["eval_state"], "identity_error")

    def test_bootstrap_mismatch_prevents_claiming_a_matched_comparison(self):
        plan = self.add_plan()
        self.evidence(plan, "local")
        self.evidence(plan, "noperm", sampled_sids=["different-frozen-draw"])
        result = S.collect(self.base)
        self.assertTrue(all(r["state"] == "unmatched_inputs" and r["local_minus_noperm"] is None for r in result["matched_local_noperm"]))

    def test_vqa_frozen_sample_and_same_layout_score_damage_are_matched(self):
        for task in ("okvqa", "textvqa"):
            registered = self.add_plan(task, task=task)
            for arm in ("noperm", "local"):
                self.evidence(registered, arm)
        result = S.collect(self.base)
        self.assertTrue(result["summary"]["coverage_complete_available_scope"])
        self.assertEqual(len(result["matched_local_noperm"]), 4)
        self.assertTrue(all(row["state"] == "matched" and row["damage_metric"] == "relative_score_drop"
                            for row in result["matched_local_noperm"]))
        registered = next(row for row in S.load_runs(self.base)[0] if row["plan"]["conditions"][0]["task"] == "textvqa")
        source = next((registered["run"] / "artifacts").rglob("*__local.json"))
        value, _ = S.read_record(source)
        value["search_sample"]["sid_sha256"] = "changed"
        self.write(source, value)
        evaluation = registered["run"] / "eval" / registered["plan"]["conditions"][0]["id"] / source.name
        value, _ = S.read_record(evaluation)
        value["artifact_file_sha256"] = S.read_record(source)[1]
        self.write(evaluation, value)
        result = S.collect(self.base)
        self.assertTrue(all(row["state"] == "unmatched_inputs" for row in result["matched_local_noperm"] if row["task"] == "textvqa"))

    def test_pairs_never_mix_models_tasks_or_budgets(self):
        for name, model, task, budget in (("a", "one", "mmlu", 300), ("b", "two", "mmlu", 300),
                                           ("c", "one", "wikitext", 300), ("d", "one", "mmlu", 50)):
            plan = self.add_plan(name, budget=budget, variant="baseline" if budget == 300 else "budget", model=model, task=task)
            for arm in ("noperm", "local"):
                self.evidence(plan, arm)
        result = S.collect(self.base)
        self.assertEqual(len(result["matched_local_noperm"]), 8)
        self.assertEqual(len(result["coverage"]), 8)
        for row in result["matched_local_noperm"]:
            self.assertEqual(row["run"], row["control_run"])
            self.assertEqual(row["damage_metric"], "ppl_ratio" if row["task"] == "wikitext" else "relative_score_drop")

    def test_local_only_variant_uses_only_its_frozen_reference_noperm(self):
        baseline = self.add_plan()
        variant = self.add_plan("no_gradient", variant="no_gradient", arms=("local",), reference=baseline)
        self.evidence(baseline, "noperm")
        self.evidence(baseline, "local")
        self.evidence(variant, "local")
        result = S.collect(self.base)
        pairs = [row for row in result["matched_local_noperm"] if row["variant"] == "no_gradient"]
        self.assertTrue(pairs)
        self.assertTrue(all(row["state"] == "matched" and row["control_run"] == str(baseline[0])
                            and row["control_origin"] == "registered_reference_baseline" for row in pairs))

    def test_reference_control_rejects_changed_region_geometry(self):
        baseline = self.add_plan()
        variant = self.add_plan("fixed_columns", variant="fixed_columns", arms=("local",), reference=baseline)
        run, plan = variant
        plan["conditions"][0]["regions"][0]["page_index"] = 1
        self.write(run / "plan.json", plan)
        self.write(run / "status.json", {"plan_sha256": S.read_record(run / "plan.json")[1]})
        self.evidence(baseline, "noperm")
        self.evidence(variant, "local")
        result = S.collect(self.base)
        pairs = [row for row in result["matched_local_noperm"] if row["variant"] == "fixed_columns"]
        self.assertTrue(all(row["state"] == "unmatched_inputs" and row["local_minus_noperm"] is None for row in pairs))

    def test_registry_aliases_and_symlinks_are_deduplicated(self):
        registered = self.add_plan()
        link = self.base / "alias"
        link.symlink_to(registered[0], target_is_directory=True)
        self.registries["budget_plans.json"]["alias"] = str(link)
        self.save_registries()
        result = S.collect(self.base)
        self.assertEqual(result["summary"]["plan_count"], 1)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(len(result["summary"]["plans"][0]["registry_aliases"]), 2)

    def test_unavailable_conditions_remain_explicit_outside_completed_available_scope(self):
        available = self.add_plan()
        self.add_plan("unavailable", model="absent", unavailable=True)
        for arm in ("noperm", "local"):
            self.evidence(available, arm)
        result = S.collect(self.base)
        self.assertTrue(result["summary"]["coverage_complete_available_scope"])
        self.assertFalse(result["summary"]["matrix_complete"])
        self.assertEqual(result["summary"]["pending_available_items"], 0)
        self.assertEqual(result["summary"]["unavailable_items"], 2)
        self.assertTrue(all(row["recomputed_cost"] is None for row in result["items"] if row["availability"] == "unavailable"))

    def test_complete_mode_reports_pending_and_fresh_output_refuses_overwrite(self):
        self.add_plan()
        output = self.base / "report"
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(S.main(["--base", str(self.base), "--output", str(output), "--mode", "complete"]), 1)
        before = (output / "summary.json").read_bytes()
        with (output / "items.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["recomputed_cost"] == "" for row in rows))
        with self.assertRaises(FileExistsError):
            S.write_report(self.base, output)
        self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_import_is_standard_library_only(self):
        script = "import sys; import regionalign.reporting.campaign_costs; print('torch' in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[2],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()

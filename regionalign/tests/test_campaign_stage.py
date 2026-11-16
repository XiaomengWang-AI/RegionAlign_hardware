"""CPU-only phase sentinel checks with synthetic registered evidence files."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from regionalign.experiments import campaign_stage as S, llm as L, vqa as V


class CampaignStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "campaign"
        self.run.mkdir()
        self.output = self.run / "phases/search.json"
        regions = [{"id": "r0", "kind": "q_proj"}, {"id": "r1", "kind": "down_proj"}]
        self.condition = {"id": "condition", "model": "tiny", "format": "int8", "task": "mmlu",
            "availability": "available",
            "regions": regions, "bootstrap_seeds": {L._bootstrap_key(r["id"], 0): 10 + i
                                                       for i, r in enumerate(regions)}}
        self.plan = {"schema_version": 2, "scope": L.SCOPE,
            "conditions": [self.condition], "arms": ["noperm", "local"], "repeat_seeds": [7],
            "campaign_policy": {"scope": "regionalign_campaign_dev", "variant": "baseline"}}
        self.save_plan()
        self.backend = L
        self.runner = SimpleNamespace(configure_from_plan=mock.Mock(side_effect=self.configured),
                                      run_stage=mock.Mock(side_effect=self.finish_stage))

    def save_plan(self):
        (self.run / "plan.json").write_text(json.dumps(self.plan))

    def configured(self, _run):
        return {**self.plan, "sha256": S.M.sha256(self.run / "plan.json")}

    def common(self):
        return {"schema_version": 2, "scope": self.plan["scope"], "condition": "condition",
                "plan_sha256": S.M.sha256(self.run / "plan.json"),
                "campaign_scope": self.plan["campaign_policy"]["scope"],
                "campaign_variant": self.plan["campaign_policy"]["variant"],
                "campaign_policy_sha256": S.M.object_sha256(self.plan["campaign_policy"])}

    def items(self, shard="0/1"):
        return self.backend.shard_items(self.backend.planned_items(self.condition, self.plan), shard)

    def write_item(self, stage, item, **fields):
        path_function = self.backend._artifact_path if stage == "search" else self.backend._eval_path
        path = path_function(self.run, "condition", item["name"])
        value = {**self.common(), **{key: item[key] for key in ("region", "repeat", "seed", "arm")}, **fields}
        if stage == "evaluate":
            artifact = self.backend._artifact_path(self.run, "condition", item["name"])
            if not artifact.exists():
                self.write_item("search", item)
            value["artifact_file_sha256"] = S.M.sha256(artifact)
        S.M.write_json_once(path, value)
        return path

    def finish_stage(self, _run, stage, condition, *, shard, device):
        self.assertEqual(condition, "condition")
        self.assertEqual(device, "cpu")
        for item in self.items(shard):
            path_function = self.backend._artifact_path if stage == "search" else self.backend._eval_path
            if not path_function(self.run, condition, item["name"]).exists():
                self.write_item(stage, item)

    def invoke(self, stage="search", shard="0/1"):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(S.C, "output_path", side_effect=lambda path: Path(path).resolve()))
            stack.enter_context(mock.patch.object(S, "_runner", return_value=self.runner))
            stack.enter_context(mock.patch.object(S, "_backend", return_value=self.backend))
            return S.run_stage(self.run, stage, "condition", shard, self.output, device="cpu")

    def vqa(self):
        self.backend = V
        self.plan["scope"] = V.SCOPE
        self.condition["task"] = "textvqa"
        self.save_plan()

    def preflight(self, **changes):
        first = self.items()[0]
        path = V._preflight_path(self.run, "condition")
        value = {**self.common(), "status": "complete", "gradient_finite": True,
                 "matched_region": first["region"], "repeat": first["repeat"],
                 "matched_arms": list(V.ARMS),
                 "arms": {arm: {"logits_allclose": True} for arm in V.ARMS}, **changes}
        S.M.write_json_once(path, value)
        return path

    def test_runner_dispatches_baseline_budget_and_five_variants(self):
        for variant in ("baseline", "budget", *sorted(S.VARIANTS)):
            with self.subTest(variant=variant), mock.patch.object(S.importlib, "import_module") as load:
                S._runner({"campaign_policy": {"variant": variant}})
            suffix = "campaign_protocol" if variant in ("baseline", "budget") else "campaign_variants"
            load.assert_called_once_with(f"regionalign.experiments.{suffix}")
        with self.assertRaises(ValueError):
            S._runner({"campaign_policy": {"variant": "unknown"}})

    def test_partial_search_resumes_and_keeps_errors_separate_from_coverage(self):
        first = self.items("0/2")[0]
        path = self.write_item("search", first, error="retained synthetic algorithm failure")
        before = path.read_bytes()
        result = self.invoke(shard="0/2")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["inventory"]["expected_count"], 2)
        self.assertEqual(result["inventory"]["present_count"], 2)
        self.assertEqual(result["inventory"]["statistics"]["error_files"], 1)
        self.assertEqual(result["inventory"]["statistics"]["status_counts"], {"artifact_present": 1, "search_error": 1})
        self.assertEqual(path.read_bytes(), before)
        self.runner.run_stage.assert_called_once()
        sentinel_before = self.output.read_bytes()
        self.assertEqual(self.invoke(shard="0/2"), result)
        self.assertEqual(self.output.read_bytes(), sentinel_before)
        self.runner.run_stage.assert_called_once()

    def test_complete_existing_phase_seals_without_calling_driver(self):
        for item in self.items():
            self.write_item("search", item)
        result = self.invoke()
        self.assertEqual(result["execution"], "existing_registered_files")
        self.runner.run_stage.assert_not_called()

    def test_evaluation_retains_rejections_invalid_budgets_and_gate_failures(self):
        items = self.items()
        self.write_item("evaluate", items[0], status="complete", rearranged={"gate": True},
                        results={"1": {"valid": True}, "2": {"valid": False}})
        self.write_item("evaluate", items[1], status="artifact_error", artifact_error="rejected", results={})
        self.write_item("evaluate", items[2], status="complete", rearranged={"gate": False},
                        results={"1": {"valid": False}})
        self.write_item("evaluate", items[3], status="evaluation_error", error="retained failure")
        result = self.invoke(stage="evaluate")
        stats = result["inventory"]["statistics"]
        self.assertEqual(stats["status_counts"], {"artifact_error": 1, "complete": 2, "evaluation_error": 1})
        self.assertEqual(stats["error_files"], 2)
        self.assertEqual((stats["valid_budget_results"], stats["invalid_budget_results"]), (1, 2))
        self.assertEqual((stats["clean_gate_passed_files"], stats["clean_gate_failed_files"]), (1, 1))
        self.assertTrue(all("source_artifact_sha256" in row for row in result["inventory"]["files"]))
        self.runner.run_stage.assert_not_called()

    def test_resource_failure_preserves_partial_files_and_publishes_no_sentinel(self):
        item = self.items()[0]
        def fail(*_args, **_kwargs):
            self.write_item("search", item)
            raise MemoryError("synthetic allocation failure")
        self.runner.run_stage.side_effect = fail
        with self.assertRaises(MemoryError):
            self.invoke()
        self.assertFalse(self.output.exists())
        path = self.backend._artifact_path(self.run, "condition", item["name"])
        before = path.read_bytes()
        self.runner.run_stage.side_effect = self.finish_stage
        self.assertEqual(self.invoke()["status"], "complete")
        self.assertEqual(path.read_bytes(), before)

    def test_missing_registered_files_prevent_publication(self):
        self.runner.run_stage.side_effect = lambda *_args, **_kwargs: None
        with self.assertRaisesRegex(RuntimeError, "phase is incomplete"):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_identity_mismatch_is_rejected_before_driver_resume(self):
        path = self.write_item("search", self.items()[0])
        data = S.M.read_json(path)
        data["repeat"] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.invoke()
        self.runner.run_stage.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_sentinel_rechecks_child_file_hashes_and_never_overwrites(self):
        self.invoke()
        before = self.output.read_bytes()
        child = self.backend._artifact_path(self.run, "condition", self.items()[0]["name"])
        child.write_bytes(child.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "existing sentinel"):
            self.invoke()
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.runner.run_stage.call_count, 1)

    def test_existing_sentinel_rejects_its_own_wrong_source_identity(self):
        self.invoke()
        sentinel = S.M.read_json(self.output)
        sentinel["wrapper_source_sha256"] = "0" * 64
        self.output.write_text(json.dumps(sentinel))
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "existing sentinel"):
            self.invoke()
        self.assertEqual(self.output.read_bytes(), before)

    def test_successful_preflight_can_be_sealed_without_rerunning_it(self):
        self.vqa()
        self.preflight()
        result = self.invoke(stage="preflight")
        self.assertEqual(result["inventory"]["expected_count"], 1)
        self.assertEqual(result["inventory"]["files"][0]["identity"]["matched_arms"], list(V.ARMS))
        self.runner.run_stage.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unsharded VQA"):
            self.invoke(stage="preflight", shard="1/2")

    def test_failed_preflight_qualification_prevents_sentinel(self):
        self.vqa()
        self.preflight(gradient_finite=False)
        with self.assertRaisesRegex(ValueError, "qualification is not successful"):
            self.invoke(stage="preflight")
        self.assertFalse(self.output.exists())
        self.runner.run_stage.assert_not_called()

    def test_evaluation_must_still_bind_its_source_artifact(self):
        item = self.items()[0]
        self.write_item("evaluate", item, status="artifact_error", artifact_error="retained")
        artifact = self.backend._artifact_path(self.run, "condition", item["name"])
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "binds its source artifact"):
            self.invoke(stage="evaluate")
        self.assertFalse(self.output.exists())

    def test_plan_change_during_phase_prevents_publication(self):
        def change(*args, **kwargs):
            self.finish_stage(*args, **kwargs)
            self.plan["note"] = "changed during execution"
            self.save_plan()
        self.runner.run_stage.side_effect = change
        with self.assertRaisesRegex(RuntimeError, "changed during the phase"):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_empty_shard_is_not_presented_as_a_completed_experiment(self):
        with self.assertRaisesRegex(ValueError, "empty phase shard"):
            self.invoke(shard="3/4")
        self.assertFalse(self.output.exists())
        self.runner.run_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()

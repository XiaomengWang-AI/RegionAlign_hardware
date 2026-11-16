"""CPU evidence checks for inherited panels and process-local campaign policy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from regionalign import config as C
from regionalign.experiments import campaign_protocol as S
from regionalign.experiments import clean_benchmarks as B
from regionalign.experiments import clean_mmlu as M
from regionalign.experiments import llm as L
from regionalign.experiments import vqa as V
from regionalign.harness import region as R


class CampaignProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.results = mock.patch.object(C, "RESULTS_ROOT", self.root)
        self.results.start()
        self.addCleanup(self.results.stop)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(S._reset_configuration)

    def parent(self, family="llm"):
        root = self.root / f"parent_{family}"
        root.mkdir()
        source = root / "source" / "historical.py"
        source.parent.mkdir()
        source.write_bytes(b"# old source, deliberately different from live code\n")
        panels = {}
        for split in ("search", "dev"):
            p = root / "panels" / "task" / f"{split}.bin"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00frozen-" + split.encode() + b"\xff\x00")
            panels[split] = {"path": str(p.relative_to(root)), "sha256": M.sha256(p),
                             "count": 4}
        condition = {"id": "model__int4__task", "model": "model", "format": "int4",
                     "task": "task", "panels": panels, "regions": [{"id": "region"}],
                     "checkpoint_manifest_sha256": "checkpoint_identity",
                     "bootstrap_seeds": {"region__r0": 123},
                     "search_bootstraps": [{"repeat": 0, "sids": ["same", "same"]}]}
        plan = {"schema_version": 2, "scope": f"{family}_regionalign_dev",
                "development_only": True, "splits": ["search", "dev"],
                "conditions": [condition], "models": {"model": {"identity": "preserved"}},
                "datasets": {"identity": "preserved"}, "search_budget": 300,
                "work_items_per_condition": 3, "arms": ["noperm", "local", "random"],
                "repeat_seeds": [1], "search_k": list(range(1, 17)),
                "evaluation_k": [1, 2, 4, 8, 16], "clean_gate_nll": 0.01,
                "sources": [{"path": "historical_missing_live.py",
                             "snapshot": "source/historical.py", "sha256": M.sha256(source)}]}
        if family == "llm":
            plan["execution_protocol"] = L._execution_protocol(1, 1, 1, 40.0)
        else:
            plan["execution"] = {"search_batch_size": 1, "grad_batch_size": 1,
                                 "eval_batch_size": 1, "min_free_gpu_gib": 40.0,
                                 "min_free_policy": V.MIN_FREE_POLICY,
                                 "real_gpu_preflight_required": True}
        M.write_json_once(root / "plan.json", plan)
        M.write_json_once(root / "status.json", {"development_only": True,
                                                "plan_sha256": M.sha256(root / "plan.json")})
        return root, plan

    def prepare(self, family="llm", **kwargs):
        parent, old = self.parent(family)
        output = S.prepare(parent, self.root / "campaign", **kwargs)
        return parent, old, output, M.read_json(output / "plan.json")

    def test_fork_preserves_registered_inputs_and_freezes_actual_sources(self):
        parent, original, output, plan = self.prepare()
        self.assertEqual(plan["conditions"], original["conditions"])
        self.assertEqual(plan["datasets"], original["datasets"])
        self.assertEqual(plan["models"], original["models"])
        self.assertEqual(plan["campaign_policy"]["parent_plan_sha256"], M.sha256(parent / "plan.json"))
        for record in plan["conditions"][0]["panels"].values():
            self.assertEqual((output / record["path"]).read_bytes(), (parent / record["path"]).read_bytes())
            self.assertNotEqual((output / record["path"]).stat().st_ino,
                                (parent / record["path"]).stat().st_ino)
        self.assertIn(str(S.MODULE_SOURCE), [s["path"] for s in plan["sources"]])
        _, loaded = L.load_plan(output)
        self.assertEqual(loaded["campaign_policy"]["scope"], S.CAMPAIGN_SCOPE)
        with self.assertRaises(FileExistsError):
            S.prepare(parent, output)

    def test_changed_parent_panel_is_rejected_before_a_new_run_exists(self):
        parent, original = self.parent()
        panel = parent / original["conditions"][0]["panels"]["search"]["path"]
        panel.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "frozen panel changed"):
            S.prepare(parent, self.root / "campaign")
        self.assertFalse((self.root / "campaign").exists())

    def test_saved_historical_source_is_checked_without_live_source_replay(self):
        parent, _ = self.parent()
        (parent / "source/historical.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source snapshot changed"):
            S.prepare(parent, self.root / "campaign")

    def test_budget_variant_uses_an_independent_plan_and_the_existing_validator(self):
        _, _, output, plan = self.prepare(variant="budget", budget=50)
        configured = S.configure_from_plan(output)
        self.assertEqual(configured["search_budget"], 50)
        self.assertEqual(plan["campaign_policy"]["parameters"], {"search_budget": 50})
        self.assertTrue(R.budget_reasons({"spent": 51, "calls": {"eval": 51, "grad": 0}},
                                        configured["search_budget"]))
        self.assertEqual(R.BUDGET, 300)

    def test_baseline_cannot_silently_change_the_compute_budget(self):
        parent, _ = self.parent()
        with self.assertRaises(ValueError):
            S.prepare(parent, self.root / "campaign", budget=50)

    def test_resource_threshold_is_applied_only_by_explicit_configuration(self):
        before = (V.MIN_FREE_GPU_GIB, copy.deepcopy(B.SETTINGS))
        _, _, output, _ = self.prepare("vqa", min_free_gpu_gib=22.0)
        self.assertEqual((V.MIN_FREE_GPU_GIB, B.SETTINGS), before)
        with self.assertRaises(ValueError):
            V.load_plan(output)
        plan = S.configure_from_plan(output)
        self.assertEqual(V.MIN_FREE_GPU_GIB, 22.0)
        changed = {k for k in B.SETTINGS if B.SETTINGS[k] != before[1][k]}
        self.assertEqual(changed, {"min_free_gpu_gib"})
        self.assertEqual(plan["execution"]["min_free_gpu_gib"], 22.0)
        V.load_plan(output)
        S._reset_configuration()
        self.assertEqual((V.MIN_FREE_GPU_GIB, B.SETTINGS), before)

    def test_output_stamps_the_campaign_identity(self):
        _, _, output, _ = self.prepare()
        plan = S.configure_from_plan(output)
        report = output / "fake_report.json"
        M.write_json_once(report, {"scope": plan["scope"], "plan_sha256": plan["sha256"]})
        saved = M.read_json(report)
        self.assertEqual(saved["campaign_scope"], S.CAMPAIGN_SCOPE)
        self.assertEqual(saved["campaign_variant"], "baseline")

    def test_new_panel_tampering_is_rejected_even_if_status_is_rebound(self):
        _, _, output, plan = self.prepare()
        record = plan["conditions"][0]["panels"]["dev"]
        p = output / record["path"]
        p.write_bytes(b"replacement")
        record["sha256"] = M.sha256(p)
        (output / "plan.json").write_text(json.dumps(plan))
        (output / "status.json").write_text(json.dumps({"plan_sha256": M.sha256(output / "plan.json")}))
        with self.assertRaisesRegex(ValueError, "inheritance identity"):
            S.configure_from_plan(output)

    def test_llm_dispatch_uses_the_frozen_execution_parameters(self):
        _, _, output, _ = self.prepare(min_free_gpu_gib=21.0)
        with mock.patch.object(L, "run_search") as search:
            S.run_stage(output, "search", "model__int4__task", shard="1/3", device="cuda:2")
        search.assert_called_once_with(output, "model__int4__task", "1/3", "cuda:2",
                                       min_free_gpu_gib=21.0, batch_size=1, grad_batch_size=1)

    def test_different_campaigns_require_separate_processes(self):
        parent, _ = self.parent()
        first = S.prepare(parent, self.root / "first")
        second = S.prepare(parent, self.root / "second", variant="budget", budget=100)
        S.configure_from_plan(first)
        self.assertEqual(S.configure_from_plan(first)["sha256"], M.sha256(first / "plan.json"))
        with self.assertRaisesRegex(RuntimeError, "independent process"):
            S.configure_from_plan(second)


if __name__ == "__main__":
    unittest.main()

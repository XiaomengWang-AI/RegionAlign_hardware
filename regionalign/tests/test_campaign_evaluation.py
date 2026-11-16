"""Final-access and source-selection isolation for campaign follow-up evaluation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from regionalign.experiments import campaign_evaluation as E


class CampaignEvaluationTests(unittest.TestCase):
    def test_final_gate_precedes_selection_and_input_access(self):
        with patch.object(E.C, "output_path", return_value=Path("/unused")), \
             patch.object(E.M, "read_json", return_value={"scope": E.SCOPE, "schema_version": 1, "mode": "final"}), \
             patch.object(E, "load_plan") as load:
            with self.assertRaisesRegex(ValueError, "--split test --final"):
                E.run_condition("/unused", "condition", split="test")
            load.assert_not_called()

    def test_transfer_cannot_be_mislabeled_as_final(self):
        with patch.object(E.C, "output_path", return_value=Path("/unused")), \
             patch.object(E.M, "read_json", return_value={"scope": E.SCOPE, "schema_version": 1, "mode": "transfer"}), \
             patch.object(E, "load_plan") as load:
            with self.assertRaisesRegex(ValueError, "--split dev"):
                E.run_condition("/unused", "condition", split="test", final=True)
            load.assert_not_called()

    def test_unknown_mode_cannot_reach_final_input_loader(self):
        with patch.object(E.C, "output_path", return_value=Path("/unused")), \
             patch.object(E.M, "read_json", return_value={"scope": E.SCOPE, "schema_version": 1, "mode": "unknown"}), \
             patch.object(E, "load_plan") as load:
            with self.assertRaisesRegex(ValueError, "unsupported"):
                E.run_condition("/unused", "condition")
            load.assert_not_called()

    def test_sealing_requires_development_evidence_for_every_registered_item(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_run"
            (source / "artifacts" / "condition").mkdir(parents=True)
            (source / "artifacts" / "condition" / "a.json").write_text("{}")
            plan = {"source_run": str(source), "mode": "final",
                    "conditions": [{"source_condition": "condition", "availability": "available",
                                    "items": ["a.json", "b.json"]}]}
            with patch.object(E, "load_plan", return_value=(root, plan, {}, None)):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    E.seal(root)
            self.assertFalse((root / "selection.json").exists())

    def test_selection_rejects_omission_and_duplicate_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text("{}")
            plan = {"conditions": [{"source_condition": "condition", "availability": "available",
                                    "items": ["a.json", "b.json"]}]}
            selection = {"plan_sha256": E.M.sha256(root / "plan.json"),
                         "records": [{"condition": "condition", "name": "a.json"}]}
            (root / "selection.json").write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "omits or duplicates"):
                E.verify_selection(root, plan)

    def test_sealed_artifact_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("plan.json", "artifact.json", "dev.json"):
                (root / name).write_text("{}")
            plan = {"conditions": [{"source_condition": "condition", "availability": "available",
                                    "items": ["a.json"]}]}
            selection = {"plan_sha256": E.M.sha256(root / "plan.json"), "records": [{
                "condition": "condition", "name": "a.json", "artifact": str(root / "artifact.json"),
                "artifact_sha256": E.M.sha256(root / "artifact.json"),
                "development": str(root / "dev.json"), "development_sha256": E.M.sha256(root / "dev.json")}]}
            (root / "selection.json").write_text(json.dumps(selection))
            (root / "artifact.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                E.verify_selection(root, plan)

    def test_final_vqa_retains_blank_reference_in_generation(self):
        from regionalign.runtimes import vqa as V
        from regionalign.experiments import vqa as D
        old_prepare, old_metrics = V.HFVQARegionRuntime._prepare_rows, D._task_metrics
        rows = [{"sid": "blank", "reference": "", "answers": [""], "target_tokens": 0},
                {"sid": "two", "reference": "two words", "answers": ["two words"], "target_tokens": 2}]
        seen = []
        def accuracy(supplied):
            seen.extend(row["sid"] for row in supplied)
            return {"acc": .5, "nonfinite_rate": 0.0, "eos_rate": 1.0, "length_rate": 0.0,
                    "records": [{"sid": row["sid"], "score": .5} for row in supplied]}
        runtime = SimpleNamespace(nll=lambda supplied, bs: [3.0 for _ in supplied], accuracy=accuracy)
        with E.final_vqa_adapter():
            report = D._task_metrics(runtime, rows)
        self.assertEqual(seen, ["blank", "two"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["nll_unavailable_samples"], 1)
        self.assertEqual(report["target_tokens"], 2)
        self.assertEqual(report["nll"], 3.0)
        self.assertIs(V.HFVQARegionRuntime._prepare_rows, old_prepare)
        self.assertIs(D._task_metrics, old_metrics)


if __name__ == "__main__":
    unittest.main()

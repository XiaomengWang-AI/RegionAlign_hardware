"""Real tiny-model transfer checks; all plans and evidence live in temporary files.

The fixture freezes a legal nonidentity layout and one legal bit directly. It
never searches, computes gradients, loads checkpoints, or opens final panels.
"""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from regionalign.algorithm import search as A
from regionalign.experiments import campaign_evaluation as E, llm as L
from regionalign.harness import core as P, region as R
from regionalign.runtimes.text import FormalTextRuntime
from regionalign.tests.helpers import TINY_GROUP, tiny_lm


class CampaignFollowupTests(unittest.TestCase):
    def fixture(self, root, fmt, source_task, rejected=False):
        """Build schema-2 source evidence and a target runtime without a search."""
        target_task = E.TARGET_TASK[source_task]
        model = tiny_lm(torch.float32)
        P.quantize(model, fmt, group=TINY_GROUP)
        rows = [{"sid": f"target-{index}", "prompt_ids": [1, 10 + index, 12],
                 "answer_ids": [5 + index] if target_task == "mmlu" else [5 + index, 7],
                 "choice_token_ids": [5, 6, 7, 8], "answer": 0,
                 "target_tokens": 1 if target_task == "mmlu" else 2}
                for index in range(2)]
        L._normalise_objective_weights(rows)
        runtime = FormalTextRuntime(model, SimpleNamespace(eos_token_id=2), {"dev": rows}, "cpu")
        runtime.task = target_task
        if target_task == "mmlu":
            for row, logits in zip(rows, runtime.prefill_logits(rows)):
                row["answer"] = int(logits[row["choice_token_ids"]].argmax())
                row["answer_ids"] = [row["choice_token_ids"][row["answer"]]]
                runtime._answers[row["sid"]] = torch.tensor(row["answer_ids"], dtype=torch.long)
        dims = P.dims(model)
        region = R.make_region("gate_proj", 0, 0, dims, bits=P.FORMAT_BITS[fmt], page_bytes=64)
        permutation = R.placement(region, dims, 2 * TINY_GROUP)
        self.assertFalse(P.is_identity(permutation))
        fingerprint = R.model_checksum(model)
        P.apply_permutation(runtime, permutation)
        try:
            # Select by legality alone, never by task loss or damage.
            flips = next([[region["tensor"], element, 0]] for element in range(16)
                         if R.check_region_attack(runtime, region,
                             [[region["tensor"], element, 0]], 1)["valid"])
        finally:
            P.apply_permutation(runtime, permutation, inverse=True)
        self.assertEqual(R.model_checksum(model), fingerprint)
        source_id, target_id = f"tiny__{fmt}__{source_task}", f"tiny__{fmt}__{target_task}"
        condition = {"id": source_id, "model": "tiny", "format": fmt, "task": source_task,
                     "dims": dims, "regions": [region],
                     "bootstrap_seeds": {L._bootstrap_key(region["id"], 0): 17},
                     "panels": {"search": {"count": 3}}}
        target = {**condition, "id": target_id, "task": target_task,
                  "panels": {"search": {"count": 2}}}
        source_run = root / "source"
        source_run.mkdir()
        source = {"schema_version": 2, "scope": L.SCOPE, "conditions": [condition, target],
                  "repeat_seeds": [7], "arms": ["local"], "search_budget": R.BUDGET,
                  "search_k": list(R.K_SCHEDULE), "evaluation_k": list(L.EVAL_K),
                  "clean_gate_nll": R.GATE_NLL, "mmlu_score_gate": L.MMLU_SCORE_GATE,
                  "execution_protocol": L._execution_protocol(min_free_gpu_gib=0)}
        E.M.write_json_once(source_run / "plan.json", source)
        source["sha256"] = E.M.sha256(source_run / "plan.json")
        item = L.planned_items(condition, source)[0]
        search_rows = [{"sid": f"frozen-source-{index}", "target_tokens": index + 1}
                       for index in range(3)]
        _, bootstrap = L._bootstrap_rows(search_rows, item["bootstrap_seed"])
        artifact = {"schema_version": 2, "scope": L.SCOPE, "condition": source_id,
                    "model": "tiny", "format": fmt, "task": source_task,
                    "region": region, "repeat": 0, "seed": 7, "arm": "local",
                    "bootstrap_seed": item["bootstrap_seed"], "bootstrap": bootstrap,
                    "execution_protocol": source["execution_protocol"],
                    "plan_sha256": source["sha256"], "search_rows": 3,
                    "quantization": P.quantization_audit(model), "quantization_schema_version": 2,
                    "permutation": P.spec_to_json(permutation), "attacks": {"1": flips},
                    "spent": 1, "calls": {"eval": 1, "grad": 0}}
        if rejected:
            # A naive target-task rewrite would make this invalid source identity
            # appear valid. The first validation must preserve its rejection.
            artifact["task"] = target_task
        artifact_path = source_run / "artifacts" / source_id / item["name"]
        E.M.write_json_once(artifact_path, artifact)
        development = source_run / "eval" / source_id / item["name"]
        E.M.write_json_once(development, {"plan_sha256": source["sha256"],
            "artifact_file_sha256": E.M.sha256(artifact_path),
            "status": "artifact_error" if rejected else "complete"})
        run = root / "transfer"
        run.mkdir()
        plan = {"schema_version": 1, "scope": E.SCOPE, "mode": "transfer",
                "source_run": str(source_run), "source_plan_sha256": source["sha256"],
                "sources": [], "inputs": {}, "prior_clean_final_exposure": False,
                "conditions": [{"source_condition": source_id, "target_condition": target_id,
                    "model": "tiny", "format": fmt, "source_task": source_task,
                    "target_task": target_task, "availability": "available", "items": [item["name"]]}]}
        E.M.write_json_once(run / "plan.json", plan)
        selection = {"scope": E.SCOPE, "mode": "transfer",
                     "plan_sha256": E.M.sha256(run / "plan.json"), "records": [{
                         "condition": source_id, "name": item["name"], "artifact": str(artifact_path),
                         "artifact_sha256": E.M.sha256(artifact_path), "development": str(development),
                         "development_sha256": E.M.sha256(development)}]}
        E.M.write_json_once(run / "selection.json", selection)
        return SimpleNamespace(runtime=runtime, rows=rows, source=source, condition=condition,
            target=target, item=item, artifact=artifact, artifact_path=artifact_path,
            artifact_bytes=artifact_path.read_bytes(), run=run, plan=plan, fingerprint=fingerprint,
            output=run / "eval" / source_id / item["name"])

    def run_transfer(self, fixture, mutate_during_load=False, distinguish_baselines=False):
        real_metrics = L._task_metrics
        metrics_seen, checked_tasks = [], []
        real_check = L._artifact_reasons

        def metrics(runtime, rows, task, batch_size):
            metrics_seen.append(task)
            result = real_metrics(runtime, rows, task, batch_size)
            if distinguish_baselines and len(metrics_seen) == 1:
                # Simulate a permitted 0.002-nat numerical clean drift. This
                # distinguishes the original diagnostic from the deployed layout.
                result["nll"] += 0.002
                result["ppl"] = math.exp(result["nll"])
            return result

        def check(runtime, artifact, condition, item, plan):
            checked_tasks.append((artifact["task"], condition["task"]))
            return real_check(runtime, artifact, condition, item, plan)

        def load(_run, _plan, condition, split, device, minimum):
            self.assertEqual(condition["id"], fixture.target["id"])
            self.assertEqual((split, device, minimum), ("dev", "cpu", 0.0))
            if mutate_during_load:
                fixture.artifact_path.write_bytes(fixture.artifact_bytes + b"\n")
            return fixture.runtime, fixture.rows, fixture.runtime.quantization, {}

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(E.C, "output_path", side_effect=lambda path: Path(path)))
            stack.enter_context(mock.patch.object(E, "load_plan",
                return_value=(fixture.run, fixture.plan, fixture.source, L)))
            stack.enter_context(mock.patch.object(L, "_load_runtime", side_effect=load))
            stack.enter_context(mock.patch.object(L, "_task_metrics", side_effect=metrics))
            stack.enter_context(mock.patch.object(L, "_artifact_reasons", side_effect=check))
            searches = stack.enter_context(mock.patch.object(A, "search", side_effect=AssertionError("no re-search")))
            gradients = stack.enter_context(mock.patch.object(fixture.runtime, "grads", side_effect=AssertionError("no gradients")))
            final_inputs = stack.enter_context(mock.patch.object(E, "final_text_rows", side_effect=AssertionError("no final inputs")))
            final_vqa = stack.enter_context(mock.patch.object(E, "final_vqa_rows", side_effect=AssertionError("no final inputs")))
            final_access = stack.enter_context(mock.patch.object(E, "final_access", side_effect=AssertionError("no final access")))
            stack.enter_context(mock.patch.object(L, "_search_module", side_effect=AssertionError("no search module")))
            E.run_condition(fixture.run, fixture.condition["id"], device="cpu", split="dev")
            for operation in (searches, gradients, final_inputs, final_vqa, final_access):
                operation.assert_not_called()
        return metrics_seen, checked_tasks

    def assert_no_private_or_tensor_values(self, value):
        self.assertFalse(torch.is_tensor(value))
        if isinstance(value, dict):
            self.assertNotIn("_logits", value)
            for child in value.values():
                self.assert_no_private_or_tensor_values(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_no_private_or_tensor_values(child)

    def test_real_integer_runtime_transfers_in_both_directions_without_research(self):
        for fmt in ("int8", "int4"):
            for task in ("mmlu", "wikitext"):
                with self.subTest(format=fmt, source_task=task), tempfile.TemporaryDirectory() as directory:
                    fixture = self.fixture(Path(directory), fmt, task)
                    metrics, checks = self.run_transfer(fixture, distinguish_baselines=True)
                    target = fixture.target["task"]
                    report = E.M.read_json(fixture.output)
                    json.dumps(report, allow_nan=False)
                    self.assert_no_private_or_tensor_values(report)
                    self.assertTrue(metrics)
                    self.assertEqual(set(metrics), {target})
                    self.assertEqual(checks[0], (task, task))
                    self.assertEqual(checks[1], (target, target))
                    self.assertEqual(fixture.artifact_path.read_bytes(), fixture.artifact_bytes)
                    self.assertEqual(report["artifact_file_sha256"], E.M.sha256(fixture.artifact_path))
                    self.assertEqual(report["artifact_object_sha256"], E.M.object_sha256(fixture.artifact))
                    self.assertNotEqual(report["scoring_view_object_sha256"], report["artifact_object_sha256"])
                    self.assertEqual(R.model_checksum(fixture.runtime.lm), fixture.fingerprint)
                    self.assertEqual(report["scope"], E.SCOPE)
                    self.assertEqual((report["mode"], report["split"], report["final"]), ("transfer", "dev", False))
                    self.assertEqual((report["source_task"], report["target_task"]), (task, target))
                    self.assertFalse(report["target_feedback_to_source_selection"])
                    self.assertEqual(report["status"], "complete")
                    self.assertTrue(report["rearranged"]["gate"])
                    self.assertFalse(report["rearranged"]["identity"])
                    result = report["results"]["1"]
                    self.assertTrue(result["valid"], result.get("reasons"))
                    self.assertEqual(result["damage_baseline"], "same_rearranged_unfaulted_layout")
                    expected = result["metrics"]["nll"] - report["rearranged"]["metrics"]["nll"]
                    self.assertAlmostEqual(result["delta_nll"], expected)
                    self.assertAlmostEqual(result["delta_nll"] - result["diagnostic_delta_nll_from_original_clean"], 0.002, places=5)
                    self.assertEqual(report["summary"]["valid_evaluations"], 1)
                    self.assertEqual("accuracy" in result["metrics"], target == "mmlu")
                    self.assertEqual("relative_score_drop" in result, target == "mmlu")

    def test_rejected_source_task_cannot_be_laundered_by_the_scoring_view(self):
        for fmt in ("int8", "int4"):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory), fmt, "mmlu", rejected=True)
                metrics, checks = self.run_transfer(fixture)
                report = E.M.read_json(fixture.output)
                self.assertEqual(checks[0], ("wikitext", "mmlu"))
                self.assertEqual(checks[1], ("wikitext", "wikitext"))
                self.assertEqual(report["status"], "artifact_error")
                self.assertIn("artifact task differs from plan", report["artifact_error"])
                self.assertEqual(report["results"], {})
                self.assertEqual(report["summary"]["valid_evaluations"], 0)
                self.assertFalse(report["summary"]["success_any"])
                self.assertEqual(metrics, ["wikitext"])
                self.assertEqual(fixture.artifact_path.read_bytes(), fixture.artifact_bytes)
                self.assertEqual(R.model_checksum(fixture.runtime.lm), fixture.fingerprint)
                json.dumps(report, allow_nan=False)
                self.assert_no_private_or_tensor_values(report)

    def test_artifact_drift_during_model_loading_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), "int8", "mmlu")
            with self.assertRaisesRegex(RuntimeError, "sealed source evidence changed"):
                self.run_transfer(fixture, mutate_during_load=True)
            self.assertFalse(fixture.output.exists())
            self.assertEqual(R.model_checksum(fixture.runtime.lm), fixture.fingerprint)


if __name__ == "__main__":
    unittest.main()

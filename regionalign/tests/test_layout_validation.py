"""No checkpoints or frozen inputs: exercise supplement/restoration on tiny LMs."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from regionalign.tests.helpers import TINY_GROUP, tiny_lm
from regionalign.experiments import layout_validation as V, llm as L
from regionalign.runtimes.text import FormalTextRuntime


def tiny_runtime():
    model = tiny_lm()
    V.P.quantize(model, "int8", group=TINY_GROUP)
    rows = [{"sid": f"s{i}", "prompt_ids": [3, 4, 5][:i + 1],
             "answer_ids": [6, 7], "target_tokens": 2, "objective_weight": 1.0}
            for i in range(3)]
    runtime = FormalTextRuntime(model, SimpleNamespace(eos_token_id=2),
                                {"search": rows, "dev": rows}, "cpu")
    dims = V.P.dims(model)
    regions = [V.R.make_region(kind, 0, 1, dims, page_bytes=64)
               for kind in ("q_proj", "v_proj", "gate_proj", "down_proj")]
    condition = {"id": "tiny__int8__wikitext", "model": "tiny", "format": "int8",
                 "task": "wikitext", "dims": dims, "regions": regions}
    plan = {"scope": L.SCOPE, "sha256": "frozen-plan", "clean_gate_nll": 0.01,
            "sources": [], "execution_protocol": {"min_free_gpu_gib": 0}}
    return runtime, rows, condition, plan


class LayoutValidationTests(unittest.TestCase):
    def test_layouts_are_seeded_nonidentity_and_region_bound(self):
        _, _, condition, _ = tiny_runtime()
        first = V.planned_layouts(condition, 42)
        self.assertEqual(first, V.planned_layouts(condition, 42))
        self.assertEqual(len(first), 4)
        for layout in first:
            self.assertTrue(layout["applicable"])
            self.assertFalse(V.P.is_identity(layout["permutation"]))
        with self.assertRaises(ValueError):
            V.planned_layouts(condition, 42, ["unknown-region"])

    def test_no_fault_forward_and_inverse_preserve_exact_state(self):
        runtime, rows, condition, plan = tiny_runtime()
        measure = lambda rt, rs: L._task_metrics(rt, rs, "wikitext", 1)
        original = measure(runtime, rows)
        reference = runtime.prefill_logits(rows)
        state = V.runtime_state(runtime)
        for layout in V.planned_layouts(condition, 7):
            report = V.validate_layout(runtime, rows, condition, plan, layout,
                                       original, reference, state, measure)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["inverse_state"], state)
            self.assertTrue(report["restored"]["state_matches_original"])
        self.assertEqual(V.runtime_state(runtime), state)

    def test_evaluation_oom_still_inverts_the_applied_layout(self):
        runtime, rows, condition, plan = tiny_runtime()
        state = V.runtime_state(runtime)
        layout = V.planned_layouts(condition, 2)[-1]
        with self.assertRaises(MemoryError):
            V.validate_layout(runtime, rows, condition, plan, layout,
                              {"status": "complete", "nll": 1.0}, runtime.prefill_logits(rows),
                              state, mock.Mock(side_effect=MemoryError("synthetic OOM")))
        self.assertEqual(V.runtime_state(runtime), state)

    def test_failed_inverse_is_never_reported_as_equivalence_success(self):
        runtime, rows, condition, plan = tiny_runtime()
        measure = lambda rt, rs: L._task_metrics(rt, rs, "wikitext", 1)
        state, original = V.runtime_state(runtime), measure(runtime, rows)
        apply = V.P.apply_permutation
        def corrupt_inverse(rt, spec, inverse=False):
            apply(rt, spec, inverse=inverse)
            if inverse:
                next(rt.lm.parameters()).data.reshape(-1)[0] += 1
        with mock.patch.object(V.P, "apply_permutation", side_effect=corrupt_inverse):
            with self.assertRaises(V.RestorationError) as caught:
                V.validate_layout(runtime, rows, condition, plan, V.planned_layouts(condition, 2)[0],
                                  original, runtime.prefill_logits(rows), state, measure)
        self.assertNotEqual(caught.exception.expected, caught.exception.actual)

    def test_campaign_is_only_configured_when_the_input_plan_declares_it(self):
        from regionalign.experiments import campaign_protocol
        declared = {"campaign_policy": {"resource_policy": {"min_free_gpu_gib": 20.0}}}
        with mock.patch.object(V.C, "output_path", return_value=Path("/frozen")), \
                mock.patch.object(V.M, "read_json", side_effect=[{}, declared]), \
                mock.patch.object(campaign_protocol, "configure_from_plan", return_value=declared) as configure:
            self.assertIsNone(V._configure_campaign("/frozen"))
            configure.assert_not_called()
            self.assertEqual(V._configure_campaign("/frozen"), declared["campaign_policy"])
            configure.assert_called_once_with("/frozen")

    def test_checksum_covers_visual_parameters_and_residual_order(self):
        runtime, _, _, _ = tiny_runtime()
        runtime.model = torch.nn.Module()
        runtime.model.add_module("language", runtime.lm)
        runtime.model.add_module("visual", torch.nn.Linear(2, 2))
        before = V.runtime_state(runtime)
        runtime.model.visual.weight.data[0, 0] += 1
        self.assertNotEqual(V.runtime_state(runtime), before)
        runtime.residual = torch.arange(64).flip(0)
        self.assertIsNotNone(V.runtime_state(runtime)["residual"])

    def test_gradient_qualification_records_all_eight_probes_without_state_change(self):
        runtime, rows, condition, _ = tiny_runtime()
        before = V.runtime_state(runtime)
        with tempfile.TemporaryDirectory() as raw:
            report = V.qualify_runtime(runtime, rows, condition, "cpu", Path(raw))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["registered_probes"], 8)
            self.assertEqual(len(list(Path(raw).glob("gradient_probe_*.json"))), 8)
        self.assertFalse(report["scientific_result"])
        self.assertFalse(report["covers_full_search_algorithm"])
        self.assertEqual({r["sid"] for r in report["probes"]}, {"s0", "s2"})
        self.assertEqual(V.runtime_state(runtime), before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in runtime.lm.parameters()))

    def test_gradient_oom_is_a_resource_result_and_remaining_probes_are_measured(self):
        runtime, rows, condition, _ = tiny_runtime()
        original = runtime.grads
        calls = 0
        def grads(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError("synthetic allocation failure")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(runtime, "grads", side_effect=grads):
            report = V.qualify_runtime(runtime, rows, condition, "cpu", Path(raw))
        self.assertEqual(report["status"], "resource_error")
        self.assertEqual(report["attempted_probes"], 8)
        self.assertEqual(report["completed_probes"], 7)
        self.assertEqual(report["probes"][0]["status"], "resource_error")
        self.assertTrue(all(r["status"] == "complete" for r in report["probes"][1:]))

    def test_independent_report_labels_prefix_and_never_mutates_plan(self):
        runtime, rows, condition, plan = tiny_runtime()
        backend = SimpleNamespace(load_plan=mock.Mock(return_value=(Path("/frozen"), plan)),
                                  _condition=lambda *_: condition,
                                  _load_runtime=mock.Mock(return_value=(runtime, rows, {}, {})),
                                  _task_metrics=L._task_metrics, _assert_frozen=mock.Mock())
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(V.C, "RESULTS_ROOT", Path(raw)), \
                mock.patch.object(V, "_configure_campaign", return_value=None), \
                mock.patch.object(V, "_adapter", return_value=backend):
            output, report = V.run_validation("/frozen", condition["id"], device="cpu", max_rows=2,
                                              region_ids=[condition["regions"][0]["id"]])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["panel_coverage"], "prefix_qualification")
            self.assertEqual(report["evaluated_rows"], 2)
            self.assertFalse(report["original_plan_contains_this_evaluator"])
            self.assertEqual(report["plan_sha256"], "frozen-plan")
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue(any(r["path"].endswith("layout_validation.py") for r in report["sources"]))
            self.assertEqual(plan["sources"], [])

    def test_loading_failure_gets_an_independent_classified_report(self):
        _, _, condition, plan = tiny_runtime()
        backend = SimpleNamespace(load_plan=lambda _: (Path("/frozen"), plan),
                                  _condition=lambda *_: condition,
                                  _load_runtime=mock.Mock(side_effect=MemoryError("no memory")))
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(V.C, "RESULTS_ROOT", Path(raw)), \
                mock.patch.object(V, "_configure_campaign", return_value=None), \
                mock.patch.object(V, "_adapter", return_value=backend):
            output, report = V.run_validation("/frozen", condition["id"], device="cpu")
            self.assertEqual(report["status"], "resource_error")
            self.assertTrue((output / "protocol.json").is_file())
            self.assertTrue((output / "report.json").is_file())
            self.assertEqual(report["cases"], [])

    def test_cli_has_no_final_test_or_arbitrary_split(self):
        parser = V.parser()
        commands = next(a for a in parser._actions if getattr(a, "choices", None))
        self.assertEqual(set(commands.choices), {"run", "qualify"})
        for command in commands.choices.values():
            self.assertFalse({"split", "final", "test"} & {a.dest for a in command._actions})


if __name__ == "__main__":
    unittest.main()

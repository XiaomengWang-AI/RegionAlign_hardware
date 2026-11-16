"""CPU checks that campaign restrictions reach the evaluator and search context."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from regionalign import config as C
from regionalign.experiments import campaign_protocol as B
from regionalign.experiments import campaign_variants as S
from regionalign.experiments import llm as L
from regionalign.harness import core as P, region as R
from regionalign.tests import test_campaign_protocol as fixtures
from regionalign.tests.helpers import tiny_text_runtime


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = mock.patch.object(C, "RESULTS_ROOT", self.root)
        self.results.start()
        self.addCleanup(self.results.stop)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(S._reset_configuration)

    def parent(self, family="llm"):
        root, plan = fixtures.CampaignProtocolTests.parent(self, family)
        dims = {"layers": 32, "hidden": 64, "mlp": 128, "heads": 4,
                "head_dim": 16, "group": 16}
        regions = []
        for band, layer in (("early", 0), ("middle", 15), ("late", 30)):
            for kind in ("q_proj", "v_proj", "gate_proj", "down_proj"):
                region = R.make_region(kind, layer, 0, dims, bits=4)
                region["depth_band"] = band
                regions.append(region)
        plan.update(repeat_seeds=[1, 2, 3], regions_per_condition=12,
                    work_items_per_condition=108,
                    region_design={"bands": ["early", "middle", "late"],
                                   "kinds": ["q_proj", "v_proj", "gate_proj", "down_proj"],
                                   "page_bytes": 4096})
        plan["conditions"][0].update(dims=dims, regions=regions, bootstrap_seeds={
            f"{region['id']}__r{repeat}": 100 * index + repeat
            for index, region in enumerate(regions) for repeat in range(3)})
        (root / "plan.json").write_text(json.dumps(plan, sort_keys=True))
        (root / "status.json").write_text(json.dumps({"development_only": True,
            "plan_sha256": B.M.sha256(root / "plan.json")}))
        return root, plan

    def reference(self, parent):
        root = self.root / "reference_baseline"
        if not root.exists():
            B.prepare(parent, root)
        return root

    def policy(self, variant):
        return {"variant": variant, "parameters": {"search_budget": 300, **S.PARAMETERS[variant]}}

    def test_variant_plan_preserves_baseline_source_and_records_its_own_source(self):
        parent, _ = self.parent()
        output = S.prepare(parent, self.root / "new", variant="no_gradient",
                           reference_baseline_run=self.reference(parent))
        plan = S.configure_from_plan(output)
        self.assertEqual(plan["campaign_policy"]["variant"], "no_gradient")
        self.assertEqual(plan["campaign_policy"]["parameters"]["measured_gradients"], False)
        source_names = {r["path"] for r in plan["sources"]}
        self.assertIn(str(B.MODULE_SOURCE), source_names)
        self.assertIn(str(S.MODULE_SOURCE), source_names)
        L.load_plan(output)

    def test_no_gradient_context_never_calls_the_runtime_gradient(self):
        runtime = tiny_text_runtime(fmt="int8", n=2)
        dims = P.dims(runtime.lm)
        region = R.make_region("gate_proj", 0, 0, dims, bits=8)
        context = S.context_class(R.RegionContext, self.policy("no_gradient"))(
            runtime, region, "local", 7)
        with mock.patch.object(runtime, "grads", side_effect=AssertionError("gradient executed")):
            result = context.grad()
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertEqual(context.calls, {"eval": 0, "grad": 0})
        self.assertEqual(context.spent, 0)

    def test_static_information_refresh_does_not_measure_new_information(self):
        module = SimpleNamespace(refresh=mock.Mock(side_effect=AssertionError("refreshed")))
        adapted = S.adapt_search(module, self.policy("static_information"))
        context = SimpleNamespace(grad=mock.Mock(), profile=mock.Mock())
        group = SimpleNamespace(refreshed=False)
        adapted.refresh(context, None, None, group, 300)
        self.assertTrue(group.refreshed)
        context.grad.assert_not_called()
        context.profile.assert_not_called()

    def test_distinct_element_rule_is_additional_to_existing_bit_legality(self):
        runtime = tiny_text_runtime(fmt="int8", n=2)
        region = R.make_region("gate_proj", 0, 0, P.dims(runtime.lm), bits=8)
        flips = None
        for element in range(64):
            candidate = [[region["tensor"], element, 0], [region["tensor"], element, 1]]
            if S._original_check(runtime, region, candidate, 2)["valid"]:
                flips = candidate
                break
        self.assertIsNotNone(flips)
        result = S.check_region_attack(runtime, region, flips, 2,
                                       policy=self.policy("distinct_weights"))
        self.assertFalse(result["valid"])
        self.assertTrue(any("one faulty bit" in reason for reason in result["reasons"]))

    def test_extension_does_not_make_an_invalid_base_attack_valid(self):
        runtime = tiny_text_runtime(fmt="int8", n=2)
        region = R.make_region("gate_proj", 0, 0, P.dims(runtime.lm), bits=8)
        result = S.check_region_attack(runtime, region, [], 1,
                                       policy=self.policy("distinct_weights"))
        self.assertFalse(result["valid"])
        self.assertIn("no flips", result["reasons"])

    def test_fixed_columns_rejects_only_a_down_projection_input_axis(self):
        runtime = tiny_text_runtime(fmt="int8", n=2)
        dims = P.dims(runtime.lm)
        order = list(range(dims["mlp"]))
        order[0], order[1] = order[1], order[0]
        spec = P.validate_spec({"mlp": {"0": order}}, dims)
        down = R.make_region("down_proj", 0, 0, dims, bits=8)
        gate = R.make_region("gate_proj", 0, 0, dims, bits=8)
        self.assertTrue(S.eligibility_reasons(down, [], self.policy("fixed_columns"),
                                              spec=spec, dims=dims))
        self.assertFalse(S.eligibility_reasons(gate, [], self.policy("fixed_columns"),
                                               spec=spec, dims=dims))

    def test_equal_exposure_is_in_deployed_coordinates_and_independent_of_scores(self):
        region = {"kind": "gate_proj", "tensor": "tensor", "page": 3,
                  "page_bytes": 4096, "bits": 4}
        policy = self.policy("equal_exposure")
        self.assertEqual(S.eligibility_reasons(region, [["tensor", 3 * 8192 + 4095, 0]], policy), [])
        self.assertTrue(S.eligibility_reasons(region, [["tensor", 3 * 8192 + 4096, 0]], policy))
        view = S.exposed_region(region)
        self.assertEqual((view["page"], view["page_bytes"]), (6, 2048))
        self.assertEqual(S.eligibility_reasons(view, [["tensor", 3 * 8192 + 4095, 0]], policy), [])

    def test_equal_exposure_random_keeps_the_original_seeded_layout(self):
        runtime = tiny_text_runtime(fmt="int4", n=2)
        dims = P.dims(runtime.lm)
        region = R.make_region("gate_proj", 0, 0, dims, bits=4)
        expected = R.placement(region, dims, R.random_row(region, dims, 19))
        context = S.context_class(R.RegionContext, self.policy("equal_exposure"))(
            runtime, region, "random", 19)
        self.assertEqual(P.spec_to_json(context.placement(context.forced_row)), P.spec_to_json(expected))
        self.assertEqual(sum(stop - first for _, first, stop in context.segments), 4096)
        self.assertEqual(context.region["page_bytes"], 4096)

    def test_half_page_primary_can_differ_without_changing_the_random_arm_geometry(self):
        dims = {"hidden": 4096, "mlp": 11008, "layers": 32,
                "heads": 32, "head_dim": 128}
        region = R.make_region("down_proj", 0, 1, dims, bits=4)
        half = S.exposed_region(region)
        self.assertNotEqual(R.primary_segment(region, dims)[0], R.primary_segment(half, dims)[0])

    def test_tiny_model_searches_pass_independent_variant_evaluation(self):
        runtime = tiny_text_runtime(fmt="int4", n=2)
        dims = P.dims(runtime.lm)
        parent, plan = self.parent()
        reference = self.reference(parent)
        for variant in S.VARIANTS:
            with self.subTest(variant=variant):
                output = S.prepare(parent, self.root / variant, variant=variant,
                                   reference_baseline_run=reference)
                loaded = S.configure_from_plan(output)
                kind = "down_proj" if variant == "fixed_columns" else "gate_proj"
                region = next(r for r in loaded["conditions"][0]["regions"] if r["kind"] == kind)
                context = R.RegionContext(runtime, region, "local", 7, budget=60, bs=2, grad_bs=1)
                before = R.model_checksum(runtime.lm)
                search = L._search_module(output)
                result = search.search(context)
                self.assertLessEqual(result["spent"], 60)
                self.assertEqual(R.model_checksum(runtime.lm), before)
                verdicts = context.check(P.validate_spec(result["permutation"], dims),
                    {int(k): f for k, f in result["attacks"].items() if f})
                self.assertTrue(all(v["valid"] for v in verdicts.values()))
                if variant == "no_gradient":
                    self.assertEqual(result["calls"]["grad"], 0)
                self.assertEqual(loaded["campaign_policy"]["variant"], variant)
                S._reset_configuration()

    def test_first_published_plans_register_exactly_the_items_the_driver_will_run(self):
        expected = {"no_gradient": 36, "static_information": 36, "fixed_columns": 9,
                    "equal_exposure": 108, "distinct_weights": 108}
        for family in ("llm", "vqa"):
            parent, _ = self.parent(family)
            reference = B.prepare(parent, self.root / f"reference_{family}")
            backend = B._backend(f"{family}_regionalign_dev")
            for variant, count in expected.items():
                with self.subTest(family=family, variant=variant):
                    with mock.patch.object(B.M, "write_json_once", wraps=B.M.write_json_once) as writer:
                        output = S.prepare(parent, self.root / f"{family}_{variant}", variant=variant,
                                           reference_baseline_run=reference)
                    published = [call.args[1] for call in writer.call_args_list
                                 if Path(call.args[0]).name == "plan.json"]
                    self.assertEqual(len(published), 1)
                    plan = published[0]
                    condition = plan["conditions"][0]
                    actual = backend.planned_items(condition, plan)
                    self.assertEqual(len(actual), count)
                    self.assertEqual(plan["work_items_per_condition"], count)
                    self.assertEqual(plan["application"]["expected_work_items_by_condition"][condition["id"]], count)
                    self.assertEqual(B.M.read_json(output / "status.json")["registered_searches"], count)
                    self.assertEqual(plan["application"]["registered_items_sha256"],
                                     B.M.object_sha256(S._item_registration(plan)))
                    if variant == "fixed_columns":
                        self.assertEqual({r["kind"] for r in condition["regions"]}, {"down_proj"})
                        self.assertEqual(len(condition["regions"]), 3)
                    if count < 108:
                        self.assertEqual({item["arm"] for item in actual}, {"local"})
                    S.configure_from_plan(output)
                    S._reset_configuration()

    def test_changing_driver_application_is_rejected_even_with_a_rebound_status(self):
        parent, _ = self.parent()
        output = S.prepare(parent, self.root / "new", variant="no_gradient",
                           reference_baseline_run=self.reference(parent))
        plan = B.M.read_json(output / "plan.json")
        plan["arms"] = ["noperm", "local", "random"]
        (output / "plan.json").write_text(json.dumps(plan, sort_keys=True))
        (output / "status.json").write_text(json.dumps({
            "plan_sha256": B.M.sha256(output / "plan.json")}))
        with self.assertRaisesRegex(ValueError, "actual registered driver work items"):
            S.configure_from_plan(output)


if __name__ == "__main__":
    unittest.main()

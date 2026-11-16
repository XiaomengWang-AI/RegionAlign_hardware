"""Self-contained tests for the single-injection ASR evaluator.

Tiny random int8 model only; no checkpoint, frozen panel, or GPU.
"""
import math
import random
import unittest

import torch

from regionalign.harness import core as P
from regionalign.harness import region as R
from regionalign.experiments import llm_injection as J
from regionalign.tests.helpers import TINY_GROUP, tiny_text_runtime

PAGE = 64


class WilsonInterval(unittest.TestCase):
    def test_empty_is_undefined(self):
        self.assertEqual(J._wilson(0, 0), [None, None])

    def test_bounds_contain_point_and_stay_in_unit(self):
        lo, hi = J._wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_all_success_upper_is_one_and_lower_below_one(self):
        lo, hi = J._wilson(10, 10)
        self.assertAlmostEqual(hi, 1.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0)


class SuccessRule(unittest.TestCase):
    ORIGINAL = {"status": "complete", "nll": 1.0, "accuracy": 0.6}

    def test_mmlu_half_accuracy_drop_against_layout_reference_succeeds(self):
        reference = {"status": "complete", "nll": 1.0, "accuracy": 0.6}
        faulted = {"status": "complete", "nll": 2.0, "accuracy": 0.3}
        out = J._inject_success("mmlu", faulted, reference, self.ORIGINAL)
        self.assertTrue(out["meets_success_threshold"])
        self.assertAlmostEqual(out["relative_score_drop"], 0.5)

    def test_mmlu_success_uses_layout_reference_not_original(self):
        # Reference (this layout) is slightly better than the original; the drop
        # is measured against the layout, per AGENTS.md section 7.
        reference = {"status": "complete", "nll": 1.0, "accuracy": 0.50}
        faulted = {"status": "complete", "nll": 2.0, "accuracy": 0.25}
        out = J._inject_success("mmlu", faulted, reference, self.ORIGINAL)
        self.assertAlmostEqual(out["relative_score_drop"], 0.5)
        self.assertTrue(out["meets_success_threshold"])
        self.assertAlmostEqual(out["diagnostic_relative_score_drop_from_original_clean"],
                               (0.6 - 0.25) / 0.6)

    def test_mmlu_small_drop_fails(self):
        reference = {"status": "complete", "nll": 1.0, "accuracy": 0.6}
        faulted = {"status": "complete", "nll": 1.0, "accuracy": 0.59}
        self.assertFalse(J._inject_success("mmlu", faulted, reference, self.ORIGINAL)["meets_success_threshold"])

    def test_wikitext_doubling_ppl_succeeds(self):
        reference = {"status": "complete", "nll": 2.0}
        faulted = {"status": "complete", "nll": 2.0 + math.log(2.0)}
        out = J._inject_success("wikitext", faulted, reference, {"status": "complete", "nll": 2.0})
        self.assertTrue(out["meets_success_threshold"])
        self.assertAlmostEqual(out["ppl_ratio"], 2.0, places=5)

    def test_wikitext_just_below_doubling_fails(self):
        reference = {"status": "complete", "nll": 2.0}
        faulted = {"status": "complete", "nll": 2.0 + math.log(2.0) - 1e-6}
        self.assertFalse(J._inject_success("wikitext", faulted, reference,
                                           {"status": "complete", "nll": 2.0})["meets_success_threshold"])

    def test_wikitext_catastrophic_fault_overflows_ratio_but_still_succeeds(self):
        reference = {"status": "complete", "nll": 2.0}
        faulted = {"status": "complete", "nll": 1000.0}   # exp(delta) overflows
        out = J._inject_success("wikitext", faulted, reference, {"status": "complete", "nll": 2.0})
        self.assertIsNone(out["ppl_ratio"])               # diagnostic overflowed
        self.assertTrue(out["meets_success_threshold"])   # log-space decision still fires

    def test_nonfinite_is_not_finite_and_unsuccessful(self):
        out = J._inject_success("wikitext", {"status": "nonfinite"},
                                {"status": "complete", "nll": 2.0}, {"status": "complete", "nll": 2.0})
        self.assertFalse(out["finite"])
        self.assertFalse(out["meets_success_threshold"])


class FaultSampler(unittest.TestCase):
    def setUp(self):
        self.rt = tiny_text_runtime(fmt="int8", n=2)
        self.d = P.dims(self.rt.lm)
        self.region = R.make_region("down_proj", 1, 1, self.d, page_bytes=PAGE)

    def test_kept_set_is_legal_and_dropping_is_per_element(self):
        rng = random.Random(12345)
        sampled, kept, verdict = J.sample_fault(self.rt, self.region, 8, rng)
        self.assertLessEqual(len(sampled), 8)
        self.assertTrue(set(map(tuple, kept)).issubset(set(map(tuple, sampled))))
        if kept:
            # the kept set as a whole is legal under the evaluator
            self.assertTrue(verdict["valid"])
            # and every kept element's combined bits are legal (per-element rule)
            by_element = {}
            for flip in kept:
                by_element.setdefault(flip[1], []).append(flip)
            for group in by_element.values():
                self.assertTrue(R.check_region_attack(self.rt, self.region, group, len(group))["valid"])

    def test_dropping_keeps_or_drops_whole_elements(self):
        # Every bit sharing an element is kept or dropped together, never split.
        rng = random.Random(2024)
        sampled, kept, _ = J.sample_fault(self.rt, self.region, 16, rng)
        sampled_by_el, kept_by_el = {}, {}
        for _, e, b in sampled:
            sampled_by_el.setdefault(e, set()).add(b)
        for _, e, b in kept:
            kept_by_el.setdefault(e, set()).add(b)
        for element, bits in kept_by_el.items():
            self.assertEqual(bits, sampled_by_el[element])

    def test_sampled_positions_live_on_the_region_page(self):
        tensor, bits, lo, hi = J._region_element_range(self.rt, self.region)
        rng = random.Random(7)
        sampled, _, _ = J.sample_fault(self.rt, self.region, 16, rng)
        for name, element, bit in sampled:
            self.assertEqual(name, tensor)
            self.assertTrue(lo <= element < hi)
            self.assertTrue(0 <= bit < bits)
            self.assertEqual(P.flip_byte(element, bit, bits) // self.region["page_bytes"],
                             self.region["page"])

    def test_same_seed_gives_same_fault(self):
        a = J.sample_fault(self.rt, self.region, 6, random.Random(99))[0]
        b = J.sample_fault(self.rt, self.region, 6, random.Random(99))[0]
        self.assertEqual(a, b)

    def test_empty_kept_is_reported_invalid(self):
        # A budget of zero draws nothing, so the kept set is empty and invalid.
        _, kept, verdict = J.sample_fault(self.rt, self.region, 0, random.Random(1))
        self.assertEqual(kept, [])
        self.assertFalse(verdict["valid"])


class ProtocolConstants(unittest.TestCase):
    def test_fault_law_is_the_frozen_contract(self):
        self.assertEqual(J.FAULT_LAW["in_range"], "drop_out_of_range")
        self.assertTrue(J.FAULT_LAW["no_change_counts_as_unsuccessful"])
        self.assertEqual(J.FAULT_LAW["legality_authority"], "region.check_region_attack")
        self.assertEqual(J.SUCCESS_REFERENCE, "same_rearranged_unfaulted_layout")

    def test_report_name_encodes_single_and_multi_budget(self):
        self.assertEqual(J._report_name("L15.gate_proj.p1__r0__local.json", [1]),
                         "L15.gate_proj.p1__r0__local__k1.json")
        self.assertEqual(J._report_name("L15.gate_proj.p1__r0__local.json", [1, 16]),
                         "L15.gate_proj.p1__r0__local__k1-16.json")

    def test_plan_hash_excludes_volatile_fields_and_detects_tampering(self):
        plan = {"scope": J.SCOPE, "schema_version": 1, "trials": 200, "ks": [1, 16]}
        plan["sha256"] = J._plan_hash(plan)
        # recomputing with transient fields added must not change the hash
        plan["source"] = {"anything": 1}
        plan["source_run_path"] = "/somewhere"
        self.assertEqual(J._plan_hash(plan), plan["sha256"])
        # changing a protocol field must change the hash
        tampered = {k: v for k, v in plan.items() if k not in ("source", "source_run_path")}
        tampered["trials"] = 1
        self.assertNotEqual(J._plan_hash(tampered), plan["sha256"])


if __name__ == "__main__":
    unittest.main()

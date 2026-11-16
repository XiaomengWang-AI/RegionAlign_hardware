"""Self-contained tests for the sign/high-bit-biased injection evaluator.

Tiny random int8 model only; no checkpoint, frozen panel, or GPU.  Only the
fault law differs from ``llm_injection``; the shared scoring/plan machinery is
reused from that module and tested there.
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch   # noqa: F401 - imported so a torch-less env fails loudly, like the sibling test

from regionalign.harness import core as P
from regionalign.harness import region as R
from regionalign.experiments import llm_injection as J
from regionalign.experiments import llm_injection_biased as B
from regionalign.tests.helpers import tiny_text_runtime

PAGE = 64


class BitWeights(unittest.TestCase):
    def test_weights_are_place_value_proportional(self):
        self.assertEqual(B._bit_weights(8), (1, 2, 4, 8, 16, 32, 64, 128))
        self.assertEqual(B._bit_weights(4), (1, 2, 4, 8))

    def test_every_bit_has_positive_probability_and_high_bits_dominate(self):
        # Sampling never excludes a bit (region-only contract, AGENTS.md §1),
        # but the place-value weighting makes high/sign bits the most likely.
        rng = random.Random(0)
        counts = [0] * 8
        for _ in range(20000):
            counts[B._sample_bit(8, rng)] += 1
        self.assertTrue(all(c > 0 for c in counts))        # no bit excluded
        self.assertGreater(counts[7], counts[0])           # sign bit most likely
        self.assertGreater((counts[6] + counts[7]) / sum(counts), 0.5)  # top-2 dominate


class BiasedFaultSampler(unittest.TestCase):
    def setUp(self):
        self.rt = tiny_text_runtime(fmt="int8", n=2)
        self.d = P.dims(self.rt.lm)
        self.region = R.make_region("down_proj", 1, 1, self.d, page_bytes=PAGE)

    def test_sampled_bits_stay_in_range_and_can_include_any_bit(self):
        _, bits, _, _ = J._region_element_range(self.rt, self.region)
        seen = set()
        for seed in range(40):
            sampled, _, _ = B.sample_fault(self.rt, self.region, 16, random.Random(seed))
            self.assertTrue(sampled)                   # something was drawn
            for _, _, bit in sampled:
                self.assertTrue(0 <= bit < bits)       # valid bit index, no exclusion
                seen.add(bit)
        # across many draws a low bit does occur -- the law biases, it does not restrict
        self.assertTrue(any(bit < bits - 2 for bit in seen))

    def test_sampled_positions_live_on_the_region_page(self):
        tensor, bits, lo, hi = J._region_element_range(self.rt, self.region)
        sampled, _, _ = B.sample_fault(self.rt, self.region, 16, random.Random(7))
        for name, element, bit in sampled:
            self.assertEqual(name, tensor)
            self.assertTrue(lo <= element < hi)
            self.assertTrue(0 <= bit < bits)
            self.assertEqual(P.flip_byte(element, bit, bits) // self.region["page_bytes"],
                             self.region["page"])

    def test_kept_set_is_legal_and_dropping_is_per_element(self):
        sampled, kept, verdict = B.sample_fault(self.rt, self.region, 8, random.Random(12345))
        self.assertLessEqual(len(sampled), 8)
        self.assertTrue(set(map(tuple, kept)).issubset(set(map(tuple, sampled))))
        if kept:
            self.assertTrue(verdict["valid"])
            by_element = {}
            for flip in kept:
                by_element.setdefault(flip[1], []).append(flip)
            for group in by_element.values():
                self.assertTrue(R.check_region_attack(self.rt, self.region, group, len(group))["valid"])

    def test_dropping_keeps_or_drops_whole_elements(self):
        sampled, kept, _ = B.sample_fault(self.rt, self.region, 16, random.Random(2024))
        sampled_by_el, kept_by_el = {}, {}
        for _, e, b in sampled:
            sampled_by_el.setdefault(e, set()).add(b)
        for _, e, b in kept:
            kept_by_el.setdefault(e, set()).add(b)
        for element, bits in kept_by_el.items():
            self.assertEqual(bits, sampled_by_el[element])

    def test_same_seed_gives_same_fault(self):
        a = B.sample_fault(self.rt, self.region, 6, random.Random(99))[0]
        b = B.sample_fault(self.rt, self.region, 6, random.Random(99))[0]
        self.assertEqual(a, b)

    def test_empty_kept_is_reported_invalid(self):
        _, kept, verdict = B.sample_fault(self.rt, self.region, 0, random.Random(1))
        self.assertEqual(kept, [])
        self.assertFalse(verdict["valid"])


class StatusCounting(unittest.TestCase):
    """_count_done credits both single- and multi-budget completed reports."""

    def _write(self, directory, name, status="complete", evaluation_k=(1, 16)):
        (directory / name).write_text(
            json.dumps({"status": status, "evaluation_k": list(evaluation_k)}),
            encoding="utf-8")

    def test_multi_budget_report_counts_for_each_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            item = "L00.down_proj.p0__r0__local.json"
            self._write(d, "L00.down_proj.p0__r0__local__k1-16.json", evaluation_k=(1, 16))
            # both (item, 1) and (item, 16) are covered by the one multi-budget report
            self.assertEqual(B._count_done(d, [item], [1, 16]), 2)

    def test_single_budget_reports_and_errors_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = "L00.down_proj.p0__r0__local.json"
            b = "L00.down_proj.p0__r0__random.json"
            self._write(d, "L00.down_proj.p0__r0__local__k1.json", evaluation_k=(1,))
            self._write(d, "L00.down_proj.p0__r0__local__k16.json", evaluation_k=(16,))
            self._write(d, "L00.down_proj.p0__r0__random__k1.json", evaluation_k=(1,))
            # an incomplete report and an error file must not be credited
            self._write(d, "L00.down_proj.p0__r0__random__k16.json",
                        status="injection_error", evaluation_k=(16,))
            self._write(d, "L00.down_proj.p0__r0__random__k16.json.error-X.json",
                        evaluation_k=(16,))
            # a 2-layout x 2-budget plan has 4 units; 3 are complete here
            self.assertEqual(B._count_done(d, [a, b], [1, 16]), 3)

    def test_budget_outside_the_plan_is_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            item = "L00.down_proj.p0__r0__local.json"
            self._write(d, "L00.down_proj.p0__r0__local__k8.json", evaluation_k=(8,))
            self.assertEqual(B._count_done(d, [item], [1, 16]), 0)


class ProtocolConstants(unittest.TestCase):
    def test_biased_law_is_a_distinct_frozen_contract(self):
        self.assertNotEqual(B.SCOPE, J.SCOPE)                    # a separate plan/scope
        self.assertEqual(B.FAULT_LAW["positions"],
                         "weighted_by_bit_place_value_over_region_page")
        self.assertEqual(B.FAULT_LAW["bit_weights"], "two_to_the_bit_index")
        self.assertTrue(B.FAULT_LAW["every_bit_positive_probability"])  # §1: no bit excluded
        self.assertEqual(B.FAULT_LAW["in_range"], "drop_out_of_range")
        self.assertTrue(B.FAULT_LAW["no_change_counts_as_unsuccessful"])
        self.assertEqual(B.FAULT_LAW["legality_authority"], "region.check_region_attack")
        # success rule and reference are inherited unchanged from the uniform law
        self.assertEqual(B.SUCCESS_REFERENCE, J.SUCCESS_REFERENCE)

    def test_sources_snapshot_both_this_module_and_the_reused_evaluator(self):
        own = B._own_sources()
        self.assertIn(B.SELF, own)
        self.assertIn(J.SELF, own)


if __name__ == "__main__":
    unittest.main()

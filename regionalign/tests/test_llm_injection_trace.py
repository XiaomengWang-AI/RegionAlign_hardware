"""Self-contained tests for the per-trial trace injection evaluator.

The trace evaluator adds nothing to the threat model: it reuses the two base
evaluators' fault laws, success rule and legality authority, and only records
each trial's outcome so a damage-exceedance curve can be drawn.  These tests
check that it maps to the right frozen laws, snapshots the right sources, and
classifies trials exactly as the base scorer does.  No checkpoint or GPU.
"""
import unittest

from regionalign.experiments import llm_injection as J
from regionalign.experiments import llm_injection_biased as B
from regionalign.experiments import llm_injection_trace as T


class LawSelection(unittest.TestCase):
    def test_laws_map_to_the_two_frozen_base_samplers(self):
        self.assertIs(T.LAWS["uniform"]["sample_fault"], J.sample_fault)
        self.assertIs(T.LAWS["highbias"]["sample_fault"], B.sample_fault)
        self.assertIs(T.LAWS["uniform"]["fault_law"], J.FAULT_LAW)
        self.assertIs(T.LAWS["highbias"]["fault_law"], B.FAULT_LAW)

    def test_prepare_rejects_an_unknown_law_before_touching_the_filesystem(self):
        with self.assertRaises(ValueError):
            T.prepare("no/such/source", "no/such/output", [], law="bogus")


class TrialClassification(unittest.TestCase):
    """A trace trial's class is exactly the base scorer's verdict."""

    def test_non_finite_is_a_numerical_failure(self):
        self.assertEqual(T._trial_class({"finite": False}), "numerical_failure")

    def test_finite_meeting_threshold_is_success(self):
        self.assertEqual(
            T._trial_class({"finite": True, "meets_success_threshold": True}), "success")

    def test_finite_below_threshold_is_other_unsuccessful(self):
        self.assertEqual(
            T._trial_class({"finite": True, "meets_success_threshold": False}),
            "other_unsuccessful")

    def test_classes_agree_with_the_shared_scorer(self):
        # a doubling-perplexity wikitext fault is a success under the shared rule
        import math
        faulted = {"status": "complete", "nll": 1.0 + math.log(2.0) + 1e-6}
        reference = {"nll": 1.0, "accuracy": None}
        outcome = J._inject_success("wikitext", faulted, reference, reference)
        self.assertEqual(T._trial_class(outcome), "success")


class ProtocolConstants(unittest.TestCase):
    def test_trace_is_a_distinct_scope_from_both_base_evaluators(self):
        self.assertNotEqual(T.SCOPE, J.SCOPE)
        self.assertNotEqual(T.SCOPE, B.SCOPE)

    def test_success_rule_and_reference_are_inherited_unchanged(self):
        self.assertEqual(T.SUCCESS_REFERENCE, J.SUCCESS_REFERENCE)
        self.assertEqual(T.DEFAULT_KS, J.DEFAULT_KS)
        self.assertEqual(T.DEFAULT_TRIALS, J.DEFAULT_TRIALS)

    def test_sources_snapshot_this_module_and_both_reused_evaluators(self):
        own = T._own_sources()
        self.assertIn(T.SELF, own)
        self.assertIn(J.SELF, own)
        self.assertIn(B.SELF, own)


if __name__ == "__main__":
    unittest.main()

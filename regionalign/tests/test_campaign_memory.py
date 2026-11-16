"""The memory-only wrapper must bind its implementation and preserve variants."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from regionalign import config as C
from regionalign.experiments import campaign_memory as S, campaign_protocol as B, campaign_variants as V
from regionalign.harness import region as R
from regionalign.tests import test_campaign_variants as fixtures


class MemoryPlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        patcher = mock.patch.object(C, 'RESULTS_ROOT', self.root); patcher.start(); self.addCleanup(patcher.stop)
        self.addCleanup(V._reset_configuration)

    def parent(self):
        return fixtures.VariantTests.parent(self)

    def test_new_baseline_binds_memory_implementation_and_preserves_inputs(self):
        parent, old = self.parent()
        new = S.prepare(parent, self.root / 'new')
        plan = B.configure_from_plan(new)
        self.assertEqual(plan['conditions'], old['conditions'])
        self.assertEqual(plan['search_budget'], old['search_budget'])
        self.assertEqual(plan['execution_verification'], S.POLICY)
        self.assertTrue({S.SELF, S.BOUNDED}.issubset({r['path'] for r in plan['sources']}))
        original = R.model_checksum
        with S.installed(new): self.assertIsNot(R.model_checksum, original)
        self.assertIs(R.model_checksum, original)

    def test_variant_reference_and_restrictions_survive_nested_freezing(self):
        parent, _ = self.parent()
        baseline = S.prepare(parent, self.root / 'base')
        run = S.prepare(parent, self.root / 'variant', variant='no_gradient', reference_baseline_run=baseline)
        plan = V.configure_from_plan(run)
        self.assertEqual(plan['application']['reference_baseline_run'], str(baseline))
        self.assertEqual(plan['arms'], ['local'])
        self.assertEqual(plan['work_items_per_condition'], 36)
        self.assertEqual(plan['execution_verification'], S.POLICY)
        self.assertTrue({S.SELF, S.BOUNDED,str(V.MODULE_SOURCE),str(B.MODULE_SOURCE)}.issubset({r['path'] for r in plan['sources']}))
        with S.installed(run): pass

    def test_unregistered_original_plan_cannot_silently_change_verification(self):
        parent, _ = self.parent()
        with self.assertRaisesRegex(ValueError, 'not registered'):
            with S.installed(parent): self.fail('unregistered policy accepted')

    def test_implementation_hash_is_checked_before_installation(self):
        parent, _ = self.parent(); run = S.prepare(parent, self.root / 'new')
        source = run / 'plan.json'; plan = json.loads(source.read_text())
        for record in plan['sources']:
            if record['path'] == S.BOUNDED: record['sha256'] = '0'*64
        source.write_text(json.dumps(plan))
        original = R.model_checksum
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            with S.installed(run): self.fail('wrong code accepted')
        self.assertIs(R.model_checksum, original)


if __name__ == '__main__': unittest.main()

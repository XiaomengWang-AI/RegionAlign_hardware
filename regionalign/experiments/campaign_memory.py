"""Freeze and execute an exact integer checksum memory adaptation.

New plans retain the original scientific protocol and panel bytes. Only the
position-sensitive integer fingerprint uses bounded temporary tensors; scales
keep the original floating-point reduction. Old plans/results are never edited.
This module and the implementation are included in every new source snapshot.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch
from regionalign.experiments import campaign_protocol as B

SELF = 'regionalign/experiments/campaign_memory.py'
BOUNDED = 'regionalign/experiments/bounded_checksum.py'
POLICY = {'schema_version': 1, 'operation': 'position_sensitive_integer_checksum',
          'implementation': BOUNDED, 'temporary_elements': 1048576,
          'parameter_result': 'exact_same_integer_fingerprint',
          'scale_reduction': 'original_unchanged_float64',
          'scientific_search_or_evaluation_change': False}


def prepare(parent_run, output, *, variant='baseline', budget=300,
            reference_baseline_run=None, min_free_gpu_gib=22.0):
    _, parent, _ = B._read_registered_plan(parent_run)
    backend = B._backend(parent['scope'])
    original_copy = backend._copy_sources
    original_write = B.M.write_json_once
    def freeze(run):
        records = original_copy(run)
        for relative in (SELF, BOUNDED):
            source = B.C.PROJECT / relative
            target = Path(run) / 'source' / relative.replace('/', '__')
            shutil.copyfile(source, target)
            digest = B.M.sha256(source)
            if B.M.sha256(target) != digest:
                raise RuntimeError('checksum implementation changed while freezing')
            records.append({'path': relative, 'snapshot': str(target.relative_to(run)),
                            'sha256': digest, 'bytes': source.stat().st_size})
        return records
    def publish(path, value):
        if Path(path).name == 'plan.json' and isinstance(value, dict):
            value['execution_verification'] = dict(POLICY)
            value['campaign_policy']['verification_sha256'] = B.M.object_sha256(POLICY)
        return original_write(path, value)
    with patch.object(backend, '_copy_sources', freeze), patch.object(B.M, 'write_json_once', publish):
        if variant in B.VARIANTS:
            return B.prepare(parent_run, output, variant=variant, budget=budget,
                             min_free_gpu_gib=min_free_gpu_gib,
                             note='Exact integer checksum; bounded temporary allocations; independent plan')
        from regionalign.experiments import campaign_variants as V
        return V.prepare(parent_run, output, variant=variant,
                         reference_baseline_run=reference_baseline_run,
                         min_free_gpu_gib=min_free_gpu_gib,
                         note='Exact integer checksum; bounded temporary allocations; independent plan')


@contextmanager
def installed(run):
    run = B.C.output_path(run)
    plan = B.M.read_json(run / 'plan.json')
    if plan.get('scope') == 'regionalign_campaign_evaluation_v1':
        source_run = B.C.output_path(plan['source_run'])
        if B.M.sha256(source_run / 'plan.json') != plan['source_plan_sha256']:
            raise ValueError('followup source plan changed')
        plan = B.M.read_json(source_run / 'plan.json')
    if (plan.get('execution_verification') != POLICY
            or plan.get('campaign_policy', {}).get('verification_sha256') != B.M.object_sha256(POLICY)):
        raise ValueError('bounded checksum is not registered by this plan')
    for relative in (SELF, BOUNDED):
        records = [r for r in plan['sources'] if r['path'] == relative]
        if len(records) != 1 or B.M.sha256(B.C.PROJECT / relative) != records[0]['sha256']:
            raise ValueError('executing checksum source does not match frozen plan')
    from regionalign.experiments.bounded_checksum import install
    with install():
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--parent-run', required=True); p.add_argument('--output', required=True)
    p.add_argument('--variant', default='baseline'); p.add_argument('--budget', type=int, default=300)
    p.add_argument('--reference-baseline-run'); p.add_argument('--min-free-gpu-gib', type=float, default=22.0)
    for name in ('stage', 'layout', 'followup'):
        p = sub.add_parser(name); p.add_argument('--run', required=True)
        p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv); B.C.setup_environment()
    if args.command == 'prepare':
        values = vars(args); values.pop('command'); print(prepare(**values)); return 0
    arguments = args.arguments
    if arguments[:1] == ['--']: arguments = arguments[1:]
    with installed(args.run):
        if args.command == 'stage':
            from regionalign.experiments.campaign_stage import main as execute
            return execute(['run', '--run', args.run, *arguments])
        if args.command == 'layout':
            from regionalign.experiments.layout_validation import main as execute
            operation = arguments[0]
            return execute([operation, '--run', args.run, *arguments[1:]])
        from regionalign.experiments.campaign_evaluation import main as execute
        operation = arguments[0]
        return execute([operation, '--run', args.run, *arguments[1:]])


if __name__ == '__main__':
    raise SystemExit(main())

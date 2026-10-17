import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from canonical_tools_loader_v1 import digest_file


ROOT = Path(__file__).resolve().parent


def execute(command, environment, prefix):
    prefix.with_suffix('.command').write_text(subprocess.list2cmdline(command) + '\n')
    with prefix.with_suffix('.log').open('x') as output:
        result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    prefix.with_suffix('.exit').write_text(str(result.returncode) + '\n')
    if result.returncode:
        raise RuntimeError(f'Actual exit {result.returncode}: {prefix}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--phase', choices=('accuracy', 'performance'), required=True)
    parser.add_argument('--build-version', choices=('canonical-v4-icpx',), default='canonical-v4-icpx')
    parser.add_argument('--validation-version', default='canonical-validation-v4-icpx')
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    for version in (args.build_version, args.validation_version):
        if '/' in version or version in {'.', '..'}:
            raise SystemExit('Invalid evidence version')
    gpu = os.environ.get('ZE_AFFINITY_MASK')
    if gpu not in ({'11'} if args.phase == 'accuracy' else {'7'}):
        raise SystemExit('Wrong allocated GPU: accuracy11/performance7; coordinator grant required')
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise SystemExit(f'B8-only: {name} must be 8')
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            raise SystemExit(f'Production scope requires {name}={expected}')
    artifact = ROOT / args.component
    evidence = artifact / args.validation_version
    official = artifact / args.build_version
    for name in ('build-stage-complete.json', 'source-readonly-postverify.json'):
        if not (official / name).is_file():
            raise SystemExit(f'Canonical full build/stage/postverify not complete: {name}')
    grant = json.loads(args.grant.read_text())
    snapshot = json.loads((official / 'source-snapshot.json').read_text())
    comparator_path = evidence / 'source-comparator-manifest.json'
    if not comparator_path.is_file():
        raise SystemExit('Prepare source comparator CPU-only before acquiring GPU')
    comparator = json.loads(comparator_path.read_text())
    source_manifest_path = Path(comparator['source_native_manifest'])
    if digest_file(source_manifest_path) != comparator['source_native_manifest_sha256']:
        raise SystemExit('Changed fresh SOURCE tested-native provenance')
    required = {
        'version': args.build_version, 'module': args.component, 'gpu': int(gpu),
        'phase': args.phase, 'coordinator_release': True,
        'source_digest': snapshot['source_digest'],
        'compiler_policy_sha256': digest_file(official / 'compiler-policy-verified.json'),
        'package_manifest_sha256': digest_file(official / 'source-binary-manifest.json'),
        'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
        'source_comparator_manifest_sha256': digest_file(comparator_path),
        'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
        'accuracy_authorized': args.phase == 'accuracy',
        'performance_authorized': args.phase == 'performance',
    }
    for key, expected in required.items():
        if type(grant.get(key)) is not type(expected) or grant[key] != expected:
            raise SystemExit(f'Wrong coordinator grant: {key}')
    deadline = float(grant['start_epoch'])
    if not math.isfinite(deadline) or deadline <= 0:
        raise SystemExit('Invalid coordinator grant start epoch')
    time.sleep(max(0, deadline - time.time()))
    worktree = ROOT.parent / 'pr' / f'muse-{args.component}-xe35'
    environments = {}
    for label in ('before', 'after'):
        site = evidence / 'before/site' if label == 'before' else official / 'site'
        environment = dict(os.environ,
            PYTHONPATH=f'{site}:{ROOT / "logits-vocab/framework-original/site"}:{ROOT}',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
            LD_LIBRARY_PATH='/opt/gfx-deps/profiling_tools/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
        environments[label] = environment
        execute(['python', str(ROOT / 'check_simple_canonical_package_v1.py'), args.component, label, '--build-version', args.build_version, '--validation-version', args.validation_version], environment, evidence / f'{args.phase}-package-{label}-preflight')
    if args.phase == 'accuracy':
        tests = {'memory': ('test_paged_alloc.py', 'test_write_req_to_token_pool.py'), 'position': ('test_compute_position.py',), 'logits-vocab': ('test_softcap_logits.py', 'test_vocab_mask.py')}[args.component]
        for label in ('before', 'after'):
            command = ['python', '-m', 'pytest', '-p', 'acceptance_cooldown', '-p', 'no:cacheprovider', '-q', *(str(worktree / 'tests' / test) for test in tests)]
            execute(command, environments[label], evidence / f'accuracy-original-{label}')
            execute(['python', str(ROOT / 'benchmark_simple_canonical_v1.py'), args.component, '--phase', 'accuracy', '--label', label, '--output', str(evidence / f'accuracy-production-{label}.json')], environments[label], evidence / f'accuracy-production-{label}')
    else:
        for label in ('before', 'after'):
            if (evidence / f'accuracy-original-{label}.exit').read_text().strip() != '0' or (evidence / f'accuracy-production-{label}.exit').read_text().strip() != '0':
                raise RuntimeError('Complete accuracy must pass before performance')
        for round_id in range(3):
            for run, label in (('B0', 'before'), ('A0', 'after'), ('A1', 'after'), ('B1', 'before')):
                prefix = evidence / f'performance-r{round_id}-{run}'
                execute(['python', str(ROOT / 'benchmark_simple_canonical_v1.py'), args.component, '--phase', 'performance', '--label', label, '--samples', '101', '--calls', '128', '--output', str(prefix.with_suffix('.json'))], environments[label], prefix)
                time.sleep(2)
    (evidence / f'{args.phase}-execution-complete.json').write_text(json.dumps({'phase': args.phase, 'component': args.component, 'gpu': gpu, 'actual_exit': 0, 'performance_analysis_pending': args.phase == 'performance'}, indent=2) + '\n')


if __name__ == '__main__':
    main()

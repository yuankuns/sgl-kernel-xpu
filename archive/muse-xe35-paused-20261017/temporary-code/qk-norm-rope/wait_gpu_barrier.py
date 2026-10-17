import argparse
import json
import math
import os
from pathlib import Path
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('diagnostic', 'official'), required=True)
parser.add_argument('--canonical-version', choices=('canonical-v4-icpx',), default='canonical-v4-icpx')
args = parser.parse_args()
filename = 'gpu6-barrier.json' if args.phase == 'diagnostic' else f'gpu6-official-{args.canonical_version}-barrier.json'
barrier = Path(os.environ.get('MUSE_BARRIER_FILE', root / filename))
ready = {'gpu': 6, 'cpu_lane': 2, 'cpus': '32-47', 'phase': args.phase, 'scope': 'B8/ISL4096/OSL16/TP1/EP1', 'barrier': str(barrier), 'status': 'waiting main GPU4-11 release; no GPU lock held', 'gpu2_released': json.loads((root / 'gpu2-release.json').read_text())['own_gpu2_flock_released']}
if args.phase == 'official':
    from canonical_check import local_path

    provenance = json.loads((local_path(os.environ['QK_OFFICIAL_ROOT']) / 'provenance.json').read_text())
    assert provenance['canonical_version'] == args.canonical_version
    ready['source_digest'] = provenance['ready_source_sha256']
    assert provenance['compiler_policy_proof']['verified']
    ready['canonical_version'] = args.canonical_version
    ready['compiler_policy_sha256'] = provenance['compiler_policy_proof']['compiler_policy_sha256']
    ready['package_manifest_sha256'] = provenance['canonical_manifest_sha256']
    ready['status'] = f'{args.canonical_version} checked; waiting new main GPU6 official grant; no GPU lock held'
ready_name = f'official-{args.canonical_version}-barrier-ready.json' if args.phase == 'official' else 'diagnostic-barrier-ready.json'
(root / ready_name).write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
epoch = os.environ.get('MUSE_SYNC_START_EPOCH')
if args.phase == 'official':
    assert epoch is None, 'Official v4-icpx requires main release JSON, not an unversioned epoch override'
if epoch is None:
    print(f'GPU6 ready; waiting for main 4-11 barrier: {barrier}', flush=True)
    while not barrier.exists():
        time.sleep(1)
    record = json.loads(barrier.read_text())
    assert record['gpu'] == 6 and record['coordinator_release'] is True
    if args.phase == 'official':
        assert record.get('phase', 'official') == 'official'
        assert record['canonical_version'] == args.canonical_version
        assert record['source_digest'] == ready['source_digest']
        assert record['compiler_policy_sha256'] == ready['compiler_policy_sha256']
        assert record['package_manifest_sha256'] == ready['package_manifest_sha256']
    for name, expected in {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}.items():
        assert record.get(name, expected) == expected, f'Muse scope rejected before GPU: {name}'
    epoch = record['start_epoch']
deadline = float(epoch)
assert math.isfinite(deadline) and deadline > 0
if args.phase == 'official':
    assert deadline > time.time(), 'Official barrier is stale; main must grant a fresh future epoch'
print(f'GPU6 barrier start epoch: {deadline}', flush=True)
time.sleep(max(0, deadline - time.time()))

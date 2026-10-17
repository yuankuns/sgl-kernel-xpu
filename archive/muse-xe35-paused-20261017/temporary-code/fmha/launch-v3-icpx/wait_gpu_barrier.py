import argparse
import json
import math
import os
from pathlib import Path
import time
from settings import ART, CANONICAL_VERSION, FREEZE


root = ART
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('diagnostic', 'official'), required=True)
args = parser.parse_args()
if args.phase != 'official':
    raise RuntimeError('v3-icpx launcher accepts only a new official grant')
filename = f'{CANONICAL_VERSION}/gpu5-official-barrier.json'
file_variable = 'MUSE_BARRIER_FILE' if args.phase == 'diagnostic' else 'MUSE_OFFICIAL_BARRIER_FILE'
epoch_variable = 'MUSE_SYNC_START_EPOCH' if args.phase == 'diagnostic' else 'MUSE_OFFICIAL_SYNC_START_EPOCH'
barrier = Path(os.environ.get(file_variable, root / filename))
if args.phase == 'official':
    if barrier.resolve() != (root / filename).resolve() or os.environ.get(epoch_variable) is not None:
        raise RuntimeError(f'official runtime requires the new {CANONICAL_VERSION} main barrier file; old path/epoch-only overrides forbidden')
ready = {'gpu': 5, 'cpu_lane': 1, 'cpus': '16-31', 'phase': args.phase,
         'scope': 'B8/ISL4096/OSL16/TP1/EP1/BF16', 'barrier': str(barrier),
         'gpu1_released': True, 'status': 'waiting main explicit GPU5 grant; performance GPUs4-7; no GPU lock held'}
(root / f'official-barrier-ready-{CANONICAL_VERSION}.json').write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
epoch = os.environ.get(epoch_variable)
if epoch is None:
    while not barrier.exists():
        time.sleep(1)
    record = json.loads(barrier.read_text())
    if record['gpu'] != 5 or record['coordinator_release'] is not True:
        raise RuntimeError('invalid main GPU5 barrier release')
    if args.phase == 'official':
        frozen = json.loads(FREEZE.read_text())
        if record.get('canonical_version') != CANONICAL_VERSION or record.get('source_digest') != frozen['source_digest']:
            raise RuntimeError('official grant does not match selected canonical frozen source')
    for key, expected in {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}.items():
        if record.get(key) != expected:
            raise RuntimeError(f'barrier {key} must be {expected}; rejected before GPU work')
    if record.get('dtype') != 'BF16':
        raise RuntimeError('official grant must explicitly preserve production BF16')
    epoch = record['start_epoch']
deadline = float(epoch)
if not math.isfinite(deadline) or deadline <= 0:
    raise RuntimeError('invalid main barrier start epoch')
print(f'GPU5 {args.phase} main explicit barrier epoch: {deadline}', flush=True)
time.sleep(max(0, deadline - time.time()))

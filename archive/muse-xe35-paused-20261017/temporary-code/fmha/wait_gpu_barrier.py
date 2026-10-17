import argparse
import json
import math
import os
from pathlib import Path
import time


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('diagnostic', 'official'), required=True)
args = parser.parse_args()
filename = 'gpu5-barrier.json' if args.phase == 'diagnostic' else 'canonical-v2/gpu5-official-barrier.json'
file_variable = 'MUSE_BARRIER_FILE' if args.phase == 'diagnostic' else 'MUSE_OFFICIAL_BARRIER_FILE'
epoch_variable = 'MUSE_SYNC_START_EPOCH' if args.phase == 'diagnostic' else 'MUSE_OFFICIAL_SYNC_START_EPOCH'
barrier = Path(os.environ.get(file_variable, root / filename))
if args.phase == 'official':
    if barrier.resolve() != (root / filename).resolve() or os.environ.get(epoch_variable) is not None:
        raise RuntimeError('official runtime requires the new canonical-v2 main barrier file; old path/epoch-only overrides forbidden')
ready = {'gpu': 5, 'cpu_lane': 1, 'cpus': '16-31', 'phase': args.phase,
         'scope': 'B8/ISL4096/OSL16/TP1/EP1/BF16', 'barrier': str(barrier),
         'gpu1_released': True, 'status': 'waiting main explicit GPU5 grant; performance GPUs4-7; no GPU lock held'}
(root / f'{args.phase}-barrier-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
epoch = os.environ.get(epoch_variable)
if epoch is None:
    while not barrier.exists():
        time.sleep(1)
    record = json.loads(barrier.read_text())
    if record['gpu'] != 5 or record['coordinator_release'] is not True:
        raise RuntimeError('invalid main GPU5 barrier release')
    if args.phase == 'official':
        frozen = json.loads((root / 'canonical-source-freeze-v2.json').read_text())
        if record.get('canonical_version') != 'canonical-v2' or record.get('source_digest') != frozen['source_digest']:
            raise RuntimeError('official grant does not match current canonical-v2 frozen source')
    for key, expected in {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}.items():
        if record.get(key, expected) != expected:
            raise RuntimeError(f'barrier {key} must be {expected}; rejected before GPU work')
    epoch = record['start_epoch']
deadline = float(epoch)
if not math.isfinite(deadline) or deadline <= 0:
    raise RuntimeError('invalid main barrier start epoch')
print(f'GPU5 {args.phase} main explicit barrier epoch: {deadline}', flush=True)
time.sleep(max(0, deadline - time.time()))

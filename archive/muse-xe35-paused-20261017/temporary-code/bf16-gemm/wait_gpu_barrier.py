import argparse
import json
import os
from pathlib import Path
import time


parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['diagnostic', 'official'], default='diagnostic')
args = parser.parse_args()
root = Path(__file__).resolve().parent
basename = 'gpu4-barrier.json' if args.phase == 'diagnostic' else 'gpu4-official-barrier.json'
barrier = Path(os.environ.get('MUSE_BARRIER_FILE', root / basename))
ready = {'gpu': 4, 'cpu_lane': 0, 'phase': args.phase,
         'scope': 'B8/ISL4096/OSL16/TP1/EP1', 'barrier': str(barrier),
         'status': 'waiting main GPU4-11 coordinator release; no GPU lock held'}
(root / (args.phase + '-barrier-ready.json')).write_text(json.dumps(ready, indent=2) + '\n')
epoch = os.environ.get('MUSE_SYNC_START_EPOCH')
print(json.dumps(ready), flush=True)
if epoch is None:
    while not barrier.exists():
        time.sleep(1)
    record = json.loads(barrier.read_text())
    assert record['gpu'] == 4 and record['coordinator_release'] is True
    epoch = record['start_epoch']
deadline = float(epoch)
assert deadline > 0
print(f'GPU4 {args.phase} barrier start epoch: {deadline}', flush=True)
time.sleep(max(0, deadline - time.time()))

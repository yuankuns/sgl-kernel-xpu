import json
import math
import os
import pathlib
import subprocess
import time

root = pathlib.Path(__file__).resolve().parent
for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
    if name in os.environ and os.environ[name] != expected:
        raise ValueError(f'{name} must be {expected} before GPU work')
full_exit = root / 'target-default-build-jobs8-link1-grf-v2.exit'
diagnostic_exit = root / 'diagnostic-barrier-round.exit'
repeat_exit = root / 'grf512-repeat-native-diagnostic.exit'
barrier = root / 'gpu9-official-barrier.json'
ready_path = root / 'official-barrier-ready.json'
record = {'gpu': 9, 'cpu_lane': 5, 'cpus': '80-95', 'waiter_pid': os.getpid(), 'gpu_lock_held': False, 'ready': False, 'official_acceptance': True, 'barrier': str(barrier), 'full_exit': str(full_exit), 'status': 'CPU-only: waiting default full normal isolated package and diagnostic drain'}
def publish():
    ready_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
publish()
while not full_exit.exists() or not diagnostic_exit.exists() or not repeat_exit.exists():
    time.sleep(5)
if int(full_exit.read_text().strip()) != 0:
    record['status'] = 'default full failed; official GPU launch prohibited'
    publish()
    raise SystemExit(1)
if int(repeat_exit.read_text().strip()) != 0:
    record['status'] = 'GRF repeat/tails failed; official synchronized readiness prohibited'
    publish()
    raise SystemExit(1)
record['ready'] = True
record['status'] = 'official full complete; waiting independent coordinator epoch, no GPU lock held'
publish()
while not barrier.exists():
    time.sleep(1)
release = json.loads(barrier.read_text())
if release['gpu'] != 9 or release['coordinator_release'] is not True or release.get('phase') != 'official':
    raise RuntimeError('invalid independent official GPU9 release')
for name, expected in {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}.items():
    if release.get(name) != expected:
        raise RuntimeError(f'official barrier {name} must be {expected}')
deadline = float(release['start_epoch'])
if not math.isfinite(deadline) or deadline <= time.time():
    raise RuntimeError('official epoch must be finite and in the future')
record['start_epoch'] = deadline
record['status'] = 'waiting independent official shared epoch'
publish()
time.sleep(max(0, deadline - time.time()))
record['actual_launch_epoch'] = time.time()
record['status'] = 'official GPU9 acceptance launched'
publish()
status = subprocess.call(['bash', str(root / 'run_acceptance_xe35.sh')])
record['exit'] = status
record['status'] = 'official acceptance terminal; actual wrapper released, logical lease retained'
publish()
raise SystemExit(status)

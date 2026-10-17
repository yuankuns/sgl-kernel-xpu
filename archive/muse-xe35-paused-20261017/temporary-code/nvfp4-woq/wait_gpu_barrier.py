import json
import math
import os
import pathlib
import subprocess
import time


root = pathlib.Path(__file__).resolve().parent
expected_scope = {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}
for name, expected in expected_scope.items():
    if name in os.environ and os.environ[name] != expected:
        raise ValueError(f'{name} must be {expected} before GPU work')
barrier = root / 'gpu9-barrier.json'
predecessor = root / 'firstpass-longcases-gpu9-native-diagnostic.exit'
ready_path = root / 'diagnostic-barrier-ready.json'
assert (root / 'native-diagnostic-export-build.exit').read_text().strip() == '0'
assert (root / 'diagnostic.READY').is_file()
ready = {
    'gpu': 9, 'cpu_lane': 5, 'cpus': '80-95', 'phase': 'diagnostic',
    'scope': 'B8/ISL4096/OSL16/TP1/EP1/BF16', 'barrier': str(barrier),
    'waiter_host_pid': os.getpid(), 'waiter_gpu_lock_held': False,
    'predecessor_session': 73413, 'predecessor_exit_file': str(predecessor),
    'predecessor_complete': predecessor.exists(), 'gpu5_released': True,
    'status': 'CPU-only waiter live; existing suite completes naturally, then shared main GPU4-11 epoch',
    'launch_command': ['bash', str(root / 'launch_native_diagnostic_xe35.sh'), 'main4-11-round1', 'production'],
    'official_acceptance': False,
}
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
while not predecessor.exists():
    time.sleep(1)
ready['predecessor_complete'] = True
ready['predecessor_exit'] = int(predecessor.read_text().strip())
ready['status'] = 'waiting main GPU4-11 coordinator release; no GPU lock held'
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
while not barrier.exists():
    time.sleep(1)
record = json.loads(barrier.read_text())
if record['gpu'] != 9 or record['coordinator_release'] is not True:
    raise RuntimeError('invalid GPU9 coordinator barrier release')
deadline = float(record['start_epoch'])
if not math.isfinite(deadline) or deadline <= 0:
    raise RuntimeError('invalid shared start_epoch')
ready['start_epoch'] = deadline
ready['status'] = 'shared epoch received; waiting until coordinator deadline'
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
print(f'GPU9 diagnostic main GPU4-11 barrier epoch: {deadline}', flush=True)
time.sleep(max(0, deadline - time.time()))
ready['actual_launcher_start_epoch'] = time.time()
ready['status'] = 'production BAAB launcher started at shared epoch'
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
status = subprocess.call(ready['launch_command'])
ready['exit'] = status
ready['status'] = 'production BAAB terminal; wrapper lock released, logical GPU9 lease retained'
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
raise SystemExit(status)

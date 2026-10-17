import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
assert sorted(os.sched_getaffinity(0)) == list(range(32, 48))
assert int((directory / 'host-launch.exit').read_text()) == 125
assert int((directory / 'maintenance-final-before-GPU.exit').read_text()) == 125
final = json.loads((directory / 'maintenance-final-before-GPU.json').read_text())
assert final['external_count'] == 1
assert final['external_or_native_compiler_processes'][0]['pid'] == 3234682
assert final['external_or_native_compiler_processes'][0]['argv'] == ['apt-get', '-qq', '-y', 'update']
known = [int((directory / name).read_text()) for name in ['host-waiter.pid', 'host-supervisor.pid']]
assert known == [3231159, 3231160]
assert all(not Path(f'/proc/{pid}').exists() for pid in known)
absent = ['container-runtime.pid', 'container-affinity.txt', 'gpu-runtime.log', 'gpu-runtime.exit',
          'maintenance-monitor.pid', 'maintenance-monitor.log', 'maintenance-during-performance.jsonl',
          'performance-maintenance-hold.json', 'performance-analysis.json', 'performance-analysis.exit']
assert all(not (directory / name).exists() for name in absent)
assert not list(directory.glob('phase-*')) and not list(directory.glob('shared-inputs-scales-*'))
ancestors = set()
ancestor = os.getpid()
while ancestor > 1:
    ancestors.add(ancestor)
    status = Path(f'/proc/{ancestor}/status').read_text()
    ancestor = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
matching = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    try:
        command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if directory.name in command and any(name in command for name in ['launch.sh', 'run_gpu.sh', 'graph_phase.py', 'maintenance_snapshot.py --watch']):
        matching.append({'pid': int(entry.name), 'command': command})
assert not matching, matching
with Path('/tmp/intel-gpu-6.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    released_epoch = time.time()
    fcntl.flock(lock, fcntl.LOCK_UN)
required = ['host-launch.exit', 'host-launch.log', 'maintenance-final-before-GPU.exit',
            'maintenance-final-before-GPU.json', 'maintenance-final-before-GPU.log',
            'main-performance-gpu6-grant-v1.json', 'grant-verified.log',
            'CPU-normal-after.json', 'CPU-normal-source.json', 'CPU-five-shape-precision-fixtures.json',
            'CPU-driver-provider.json', 'CPU-maintenance-stack.json', 'launch.sh', 'run_gpu.sh', 'graph_phase.py']
report = {'actual_handle': 29959, 'actual_host_exit': 125, 'actual_prelaunch_maintenance_exit': 125,
          'CPU_archive_evidence_complete': True, 'performance_started': False, 'GPU_wrapper_invoked': False,
          'GPU_lock_ever_acquired_by_stage': False, 'watcher_started': False,
          'hold_stage': 'last host maintenance gate before wrapper/lock/watcher, not an in-GPU watch hold',
          'hold_cause': final['external_or_native_compiler_processes'], 'hold_snapshot_epoch': final['epoch'],
          'GPU_performance_samples': 0, 'phase_shape_cases_run': 0, 'shared_actual_inputs_materialized': False,
          'warmup_events_graphs_executed': False, 'GPU6_lock_probe_free': True, 'GPU6_drain_verified_epoch': released_epoch,
          'host_owner_PIDs_absent': known, 'matching_live_processes': matching, 'expected_absent_files': absent,
          'raw_provenance_SHA256': {name: digest(directory / name) for name in required},
          'no_external_process_stopped_or_modified': True, 'no_kernel_math_precision_or_dispatch_change': True,
          'performance_acceptance': 'not run; no latency/drift/MAD or regression result',
          'stage_retired_no_restart_or_new_grant': True,
          'latest_policy': 'main prioritizes unique FM metadata normal icpx full build; GPU4–6 compile isolation; wait new main performance grant after actual build/maintenance drain',
          'routing_GPU8_and_GPU11_original_full1300': 'remain naturally running, not touched by archive'}
(directory / 'PERFORMANCE_STAGE_RETIRED_v1.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

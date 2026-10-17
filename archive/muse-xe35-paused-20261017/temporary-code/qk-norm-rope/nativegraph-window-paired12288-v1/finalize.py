import json
import os
from pathlib import Path
import subprocess
import time

from contract import ROOT, digest, read, require, save, unchanged


require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light host finalizer CPUs0–15')
unchanged()
require(read(ROOT / 'final-evidence-audit.json')['actual_exit'] == 0, 'Independent full audit required')
require((ROOT / 'final-evidence-audit.exit').read_text().strip() == '0', 'Actual audit exit')
host = int((ROOT / 'host-supervisor.pid').read_text())
require(not Path(f'/proc/{host}').exists(), 'Own host not drained')
probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], check=False)
require(probe.returncode == 0, 'PhysicalGPU6 lock not free')
exits = {name: int((ROOT / name).read_text()) for name in (
    'host-supervisor.exit', 'host-run/wrapper.exit', 'run/source.exit', 'run/port.exit', 'final-evidence-audit.exit')}
require(all(value == 0 for value in exits.values()), 'True exits not0')
save(ROOT / 'TERMINAL.json', {'actual_exit': 0, 'epoch': time.time(), 'true_exits': exits,
    'actual_supervisor_handle': 65364, 'actual_host_PID': host, 'host_starttick': 8834464,
    'container_suite_PID': 285329, 'container_suite_starttick': 8834993,
    'owned_host_PID_absent': True,
    'owned_container_PIDs_absent': read(ROOT / 'final-evidence-audit.json')['owned_container_PIDs_absent'],
    'physical_GPU6_lock_free_actual_exit': probe.returncode, 'logical_GPU6_handoff_NOT_automatic': True,
    'final_evidence_audit_sha256': digest(ROOT / 'final-evidence-audit.json'),
    'CPU_READY_sha256': digest(ROOT / 'CPU_READY.json'),
    'main_once_grant_sha256': digest(ROOT / 'main-paired-trace-grant.json'),
    'start_epoch': read(ROOT / 'main-paired-trace-grant.json')['start_epoch'],
    'performance_samples': 0, 'performance_acceptance': False,
    'old_trace_exit1_and_controlled19_unfinished_scope_preserved': True})
print(json.dumps(read(ROOT / 'TERMINAL.json'), indent=2))

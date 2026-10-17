import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from stop_full_build import processes


artifact = Path(__file__).resolve().parent
tree = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35')
archive = artifact / 'canonical-stop-v2'
archive.mkdir(exist_ok=False)
for name in ['port_full_build_v2.log', 'port_full_build_v2.exit', 'full_supervisor_v2.log',
             'full_supervisor_v2.exit', 'full_build_supervisor_v2.sh', 'build_port_xe35.sh',
             'full_supervisor_v2_scripts.sha256', 'full_build_stop_canonical.json',
             'full_build_stop_canonical.log', 'full_build_stop_canonical.exit',
             'finish_after_build.log', 'finish_after_build.exit', 'finish_before_file_rename.log',
             'finish_before_file_rename.exit']:
    shutil.copy2(artifact / name, archive / name)
shutil.copy2(tree / 'build/CMakeCache.txt', archive / 'CMakeCache.txt')
snapshot = processes()
stopped = json.loads((artifact / 'full_build_stop_canonical.json').read_text())
survivors = {pid: snapshot[int(pid)] for pid, entry in stopped['stopped_processes'].items()
             if int(pid) in snapshot and 'Z' not in snapshot[int(pid)]['state']
             and snapshot[int(pid)]['args'] == entry['args']}
assert not survivors, survivors
supervisors = {pid: entry for pid, entry in snapshot.items()
               if 'Z' not in entry['state'] and any(entry['args'] == f'bash {artifact}/{name}'
               for name in ['full_build_supervisor_v2.sh', 'finish_after_build_xe35.sh', 'run_validation_xe35.sh'])}
assert not supervisors, supervisors
lock_status = subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true']).returncode
assert lock_status == 0
waiters = {pid: entry for pid, entry in snapshot.items()
           if entry['args'] == f'python3 {artifact}/wait_gpu_barrier.py --phase long-graph'}
assert len(waiters) == 1, waiters
record = {'time_epoch': time.time(), 'full_handle': 48226, 'full_exit': 143,
          'supervisor_exit': 143, 'stop_exit': 0, 'active_full_supervisors': supervisors,
          'active_stopped_descendants': survivors, 'local_restart_authorized': False,
          'canonical_acceptance_pending': 'normal new-op full24 accuracy + BAAB',
          'no_objects_or_libraries_copied': True, 'gpu7_lock_probe_exit': lock_status,
          'gpu7_lease': 'residual; subsequent routing/shared workloads require individual main grants',
          'long_graph_handle': 96541, 'long_graph_waiters': waiters,
          'long_graph_ready': json.loads((artifact / 'long-graph-ready').read_text()),
          'performance_devices': [4, 5, 6, 7], 'accuracy_only_devices': [8, 9, 10, 11],
          'residual_perf_based_kernel_changes': [],
          'audit': 'kernel tokens equal ec29f7a modulo arch names/includes/comments; no performance-based algorithm/precision/dispatch changes',
          'deps_pin': subprocess.check_output(['git', '-C', str(artifact.parent / 'deps/sycl-tla-7b56'), 'rev-parse', 'HEAD'], text=True).strip()}
(archive / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
hashes = ''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
                 for path in sorted(archive.iterdir()) if path.is_file())
(archive / 'SHA256SUMS').write_text(hashes)
print(json.dumps(record, indent=2))

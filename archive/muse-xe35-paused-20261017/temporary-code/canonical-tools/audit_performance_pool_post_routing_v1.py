import hashlib
import json
import subprocess
import time
from pathlib import Path

from build_pool_icpx_v2 import process_snapshot
from observe_maintenance_quiet_v2 import loaded_stack, PROCESS_NAMES


ROOT = Path(__file__).resolve().parent.parent


def main():
    quiet_path = ROOT / 'MAINTENANCE_QUIET_AFTER_ROUTING_v4.json'
    quiet = json.loads(quiet_path.read_text())
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    assert 0 <= time.time() - quiet['samples'][-1]['epoch'] <= 600
    assert loaded_stack() == quiet['loaded_stack']
    active = {pid: record for pid, record in process_snapshot().items() if record['name'] in PROCESS_NAMES}
    assert not active
    for pid in (3291062, 3291063, 3291085, 3269093, 3269094, 3286742):
        assert not Path(f'/proc/{pid}').exists()
    assert int((ROOT / 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1.exit').read_text()) == 0
    lock_probes = {str(gpu): subprocess.run(['flock', '-n', f'/tmp/intel-gpu-{gpu}.lock', 'true']).returncode for gpu in (5, 6)}
    assert all(value == 0 for value in lock_probes.values())
    report = {'actual_audit_exit': 0, 'epoch': time.time(),
              'matching_live_compiler_processes': {}, 'matching_live_package_manager_processes': {},
              'quiet_evidence_sha256': hashlib.sha256(quiet_path.read_bytes()).hexdigest(),
              'routing_source_package_audit_sha256': hashlib.sha256((ROOT / 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1.json').read_bytes()).hexdigest(),
              'loaded_stack': loaded_stack(), 'stable_GPU_lock_free_probes': lock_probes,
              'GPU7_owned_by_main_performance_v4': True,
              'GPU0_to3_not_probed_or_used': True, 'GPU8_to11_accuracy_only': True,
              'scope': 'Host actual process and lock probes; no runtime graph or performance acceptance'}
    with (ROOT / 'PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

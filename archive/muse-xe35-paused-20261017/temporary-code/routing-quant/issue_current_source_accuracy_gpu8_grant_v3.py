import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    artifact = Path(__file__).resolve().parent
    directory = artifact / 'current-source-accuracy-gpu8-v3'
    ready_path = directory / 'CPU-READY.json'
    ready = json.loads(ready_path.read_text())
    assert digest(ready_path) == 'a2ba0265c66963d5aa2a3e4a403d2e8bef95d6e3e919f17da89be36470e0d874'
    assert ready['CPU_READY'] is True and ready['GPU_work_started'] is False
    assert ready['scope'] == 'original14_then_original1300_then_sixproduction_then_threefixtures'
    assert len(ready['bound_files_SHA256']) == 6613
    for name, expected in ready['bound_files_SHA256'].items():
        assert digest(Path(name)) == expected, name
    for name in ['CPU-contract-audit.exit', 'CPU-normal-package-maps-source.exit', 'CPU-original-collection.exit', 'CPU-preparation-bundle.exit', 'previous-container-drain.exit']:
        assert int((directory / name).read_text()) == 0, name
    drain_path = directory / 'previous-container-drain.json'
    drain = json.loads(drain_path.read_text())
    assert time.time() - drain['epoch'] < 120
    assert drain['container_previous_runtime_PID_absent'] == 282219
    assert drain['matching_old_runtime_processes'] == []
    assert all(value == 1 for value in drain['previous_true_failure_exits'].values())
    pids = [3312756, 3312757]
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    assert not (directory / 'host-supervisor.pid').exists()
    audit_path = artifact.parent / 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1.json'
    assert int(audit_path.with_suffix('.exit').read_text()) == 0
    stage_manifest = artifact / 'fresh-current-source-icpx-native-v3/source-native-package-manifest.json'
    assert digest(stage_manifest) == ready['current_stage_manifest_SHA256'] == 'c5ae454a738019d5f5deb28d1996d40d6c602743f190186ddd73b4cddb24f992'
    assert subprocess.run(['flock', '-n', '/tmp/intel-gpu-8.lock', 'true']).returncode == 0
    epoch = time.time()
    grant = {'issuer': 'main', 'release': True, 'grant_epoch': epoch, 'start_epoch': epoch + 120,
             'authorization': 'Explicit latest user authorization to actually issue independent main currentSOURCE v3 once GPU8 accuracy grant after fresh drain/free-lock0 and complete CPU_READY0.',
             'gpu': 8, 'accuracy_only': True, 'performance_authorized': False,
             'CPU_affinity': list(range(64, 80)), 'OMP_NUM_THREADS': 8,
             'once_only': True, 'no_automatic_retry': True, 'first_failure_save_full_tensors_and_stop': True,
             'scope': ready['scope'], 'CPU_READY_SHA256': digest(ready_path),
             'current_stage_manifest_SHA256': digest(stage_manifest),
             'main_package_audit_SHA256': digest(audit_path), 'main_package_audit_actual_exit': 0,
             'old_GPU8_host_owners_absent': pids, 'GPU8_lock_probe_free': True,
             'old_container_owner_drain_evidence_SHA256': digest(drain_path),
             'bound_files_SHA256': ready['bound_files_SHA256'], 'grant_issuer_script_SHA256': digest(Path(__file__)),
             'v1_GPU_actual_exit_preserved': 1, 'v2_CPU_actual_exit_preserved': 1,
             'cross_reference_score_bit_differences_diagnostic_only': True,
             'no_Source_native_PR_math_flags_or_tolerance_modifications': True}
    target = directory / 'main-current-source-accuracy-gpu8-grant-v3.json'
    with target.open('x') as stream:
        stream.write(json.dumps(grant, indent=2) + '\n')
    print(json.dumps({'grant_epoch': epoch, 'start_epoch': epoch + 120, 'grant_SHA256': digest(target),
                      'actual_grant_issue_exit': 0, 'GPU8_lock_free_probe': 0, 'GPU_work_started': False}))


if __name__ == '__main__':
    main()

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    evidence = artifact / 'fresh-three-source-diagnostic-v1'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    exits = {name: int((evidence / f'{name}.exit').read_text()) for name in
             ['host-launch', 'gpu-runtime', 'independent-four-component-accuracy', 'original-three-pytest']}
    assert all(status == 1 for status in exits.values())
    for name in ['package-post-fresh', 'CPU-full-fixture-audit']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    log = (evidence / 'original-three-pytest.log').read_text()
    assert 'collected 3 items' in log and '3 failed in 9.42s' in log
    known = [int((evidence / name).read_text()) for name in ['host-waiter.pid', 'host-supervisor.pid']]
    assert all(not Path(f'/proc/{pid}').exists() for pid in known)
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
        if str(evidence.name) in command and any(name in command for name in ['launch_accuracy.sh', 'run_accuracy.sh', 'accuracy_capture.py']):
            matching.append({'pid': int(entry.name), 'command': command})
    assert not matching, matching
    with Path('/tmp/intel-gpu-8.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        released_epoch = time.time()
        fcntl.flock(lock, fcntl.LOCK_UN)
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == '4cb5e929f84ebcebf98ec6702a26f1ee93190828'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    manifest = json.loads((artifact / 'fresh-source-icpx-native-v2/source-native-package-manifest.json').read_text())
    triple = 'sgl_kernel/libsgl-ops-sycl-TripleOps.so'
    assert manifest['current_install_files'][triple] == manifest['original_package_files'][triple]
    audit = json.loads((directory / 'full-fixture-audit.json').read_text())
    assert audit['CPU_evidence_complete'] and not audit['accuracy_gate_passed']
    report = {'CPU_classification_evidence_complete': True, 'original_accuracy_gate_passed': False,
              'actual_handle': 96083, 'actual_exits': exits, 'host_owner_PIDs_absent': known,
              'container_owner_PID_absent': audit['container_runtime_PID_absent'],
              'matching_live_processes': matching, 'gpu8_lock_probe_free': True,
              'GPU8_released_epoch': released_epoch, 'normal_SOURCE_post_and_full_fixture_audit_exit': 0,
              'original_three_pytest_log_SHA256': digest(evidence / 'original-three-pytest.log'),
              'grant_SHA256': digest(evidence / 'main-fresh-three-accuracy-gpu8-grant-v1.json'),
              'before_fresh_shared_TripleOps_SHA256': manifest['current_install_files'][triple],
              'shared_TripleOps_recompiled': False,
              'classification': 'fresh SOURCE separate_q fails on all three fixtures; separate_s and independently captured fused_q/s are exact; original pytest gate remains failed',
              'historical_loop_assert_stage': 'old traceback does not record iteration; new fresh fixtures do not retroactively prove old locals',
              'rootcause_limit': 'generic separate activation path is implicated; exact compiler/intermediate arithmetic mechanism not proven; no generic/BMG math changed',
              'fresh_1300_AFTER_or_performance_authorized_or_started': False,
              'source_worktree_clean': True}
    (directory / 'GPU8-actual-drain-and-classification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

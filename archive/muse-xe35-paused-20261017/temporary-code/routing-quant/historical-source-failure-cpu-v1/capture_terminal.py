import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    evidence = artifact / 'historical-fresh-source-precision-v1'
    post = artifact / 'source-precision-postcheck-v1'
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == '4cb5e929f84ebcebf98ec6702a26f1ee93190828'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    exits = {}
    for name in ['original-tests-historical', 'gpu-runtime', 'host-launch']:
        path = evidence / f'{name}.exit'
        exits[name] = int(path.read_text())
        assert exits[name] == 1
        shutil.copyfile(path, directory / path.name)
    for name in ['package-post-historical', 'package-post-fresh']:
        assert int((post / f'{name}.exit').read_text()) == 0
    assert int((post / 'CPU-result-audit.exit').read_text()) == 1
    assert int((post / 'supervisor.exit').read_text()) == 1
    original = evidence / 'original-tests-historical.log'
    archived = directory / 'historical-complete.log'
    shutil.copyfile(original, archived)
    assert digest(original) == digest(archived)
    log = archived.read_text()
    assert '3 failed, 1293 passed, 4 skipped in 2675.03s' in log and 'collected 1300 items' in log
    (directory / 'complete-failure-trace.txt').write_text(log[log.index('=================================== FAILURES'):])
    known = [3044581, 3044585, 3046276, 3068551, 3114267, 3114271]
    assert all(not Path(f'/proc/{pid}').exists() for pid in known)
    ancestors = set()
    ancestor = os.getpid()
    while ancestor > 1:
        ancestors.add(ancestor)
        status = Path(f'/proc/{ancestor}/status').read_text()
        ancestor = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    live = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if artifact.name in command and any(name in command for name in ['run_source_accuracy_only_v1.sh', 'launch_source_accuracy_only_v1.sh', 'test_per_token_group_quant_mxfp4.py']):
            live.append({'pid': int(entry.name), 'command': command})
    assert not live, live
    with Path('/tmp/intel-gpu-8.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        released_epoch = time.time()
        fcntl.flock(lock, fcntl.LOCK_UN)
    unchanged = {}
    for kind in ['historical', 'fresh']:
        before = json.loads((evidence / f'package-{kind}.json').read_text())
        after = json.loads((post / f'package-post-{kind}.json').read_text())
        for field in ['site', 'package_files_sha256', 'mapped_native_sha256', 'provenance_sha256', 'Torch_provider', 'Torch_version']:
            assert before[field] == after[field], field
        assert not after['GPU_initialized']
        unchanged[kind] = {'package_file_count': len(after['package_files_sha256']),
                           'mapped_native_count': len(after['mapped_native_sha256']),
                           'Torch_provider': after['Torch_provider'], 'Torch_version': after['Torch_version'],
                           'postcheck_sha256': digest(post / f'package-post-{kind}.json')}
    manifest = json.loads((artifact / 'fresh-source-icpx-native-v2/source-native-package-manifest.json').read_text())
    triple = 'sgl_kernel/libsgl-ops-sycl-TripleOps.so'
    assert manifest['current_install_files'][triple] == manifest['original_package_files'][triple]
    report = {'verified_epoch': time.time(), 'CPU_evidence_capture_completed': True, 'original_gate_pass': False,
              'actual_owner_handle': 24419, 'actual_exits': exits,
              'original_tests': {'collected': 1300, 'passed': 1293, 'failed': 3, 'original_skipped': 4, 'seconds': 2675.03},
              'fresh_original_1300_started': False, 'fresh_three_case_GPU_authorized': False,
              'trace_archive_sha256': digest(archived), 'known_owner_and_postwait_PIDs_absent': known,
              'matching_live_processes': live, 'gpu8_lock_nonblocking_probe_free': True, 'GPU8_released_epoch': released_epoch,
              'post_waiter_actual_exit': 1, 'post_waiter_reason': 'expected success-only audit rejected actual host-launch exit1; not overwritten',
              'normal_source_packages_post_unchanged': unchanged,
              'TripleOps_original_and_fresh_SHA_equal': manifest['current_install_files'][triple],
              'original_SKIP_reason': 'topk must not exceed the expert count; original HEAD topk4/8 with experts2, BF16/FP32',
              'failure_stages': [
                  {'node': 'test_fused_swiglu_matches_separate_dtype_surface[dtype0-0.5-2.40625]', 'line': 433,
                   'proven_stage': 'separate_q', 'mismatched_bytes': 1, 'total_packed_bytes': 16,
                   'separate_s_original_assert': 'not reached', 'fused_q_s_original_asserts': 'not reached; fused kernel was called'},
                  {'node': 'test_fused_swiglu_cpu_reference[256-dtype0]', 'line': 462,
                   'proven_stage': 'loop actual_q zero assertion; traceback does not record iteration',
                   'separate_or_fused': 'unresolved without fixture/locals; never infer first iteration',
                   'mismatched_bytes': 4, 'total_packed_bytes': 1024},
                  {'node': 'test_fused_swiglu_cpu_reference[8192-dtype0]', 'line': 462,
                   'proven_stage': 'loop actual_q zero assertion; traceback does not record iteration',
                   'separate_or_fused': 'unresolved without fixture/locals; never infer first iteration',
                   'mismatched_bytes': 119, 'total_packed_bytes': 32768}],
              'classification': 'genuine historical SOURCE original gate failure, no PORT/AFTER in this run; generic separate activation/quant coupling candidate, two loop iterations unresolved',
              'provider_drift_detected': False, 'source_worktree_clean': True,
              'fused_three_cases_all_zero_asserts_proven': False, 'six_shape_pass_does_not_replace_1300_gate': True,
              'generic_BMG_math_or_tolerance_changed': False, 'GPU_successor_or_build_launched': False}
    (directory / 'terminal-classification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

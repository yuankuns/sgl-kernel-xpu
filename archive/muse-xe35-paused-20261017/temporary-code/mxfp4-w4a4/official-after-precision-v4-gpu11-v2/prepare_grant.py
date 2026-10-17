import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    options = parser.parse_args()
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    root = artifact.parent
    source = artifact / 'historical-fresh-source-precision-v3'
    previous = root / 'SIMPLE_SOURCE_COMPILER_ACCURACY_RESULTS_v1.json'
    package = json.loads((directory / 'normal-package-check.json').read_text())
    assert int((directory / 'normal-package-check.exit').read_text()) == 0
    assert package['manifest_sha256'] == '6bb13f86e5736f1ebd3d50b59ee67c032ef44511abef71608e212c991de880ba'
    assert package['current_native_count'] == 713 and not package['xpu_initialized']
    for name in ['CPU-provider.exit', 'original-40-collect.exit', 'CPU-preflight-supervisor.exit']:
        assert int((directory / name).read_text()) == 0
    assert '40 tests collected' in (directory / 'original-40-collect.log').read_text()
    previous_results = json.loads(previous.read_text())
    assert previous_results['verified'] and previous_results['supervisor_actual_exit'] == 0
    assert previous_results['known_supervisor_and_wrapper_pids_absent'] and previous_results['gpu'] == 11
    for name in ['source-paired-precision.exit', 'original-40-historical.exit', 'original-40-fresh.exit', 'gpu-runtime.exit']:
        assert int((source / name).read_text()) == 0
    required = [previous, root / 'simple-source-compiler-accuracy-gpu11-v2-supervisor.exit',
                root / 'SIMPLE_SOURCE_COMPILER_ACCURACY_GPU11_GRANT_v2.json',
                directory / 'normal-package-check.json', directory / 'CPU-provider.json', directory / 'original-40-collect.log',
                artifact / 'canonical-v4-icpx/source-binary-manifest.json',
                artifact / 'canonical-source-freeze-v4-icpx.json',
                artifact / 'COORDINATOR_FULL_ICPX_AUDIT_v4.json',
                artifact / 'fresh-source-icpx-native-v2/cpu-native-acceptance.json',
                artifact / 'fresh-source-icpx-native-v2/source-native-package-manifest.json',
                source / 'source-paired-precision.json', source / 'source-site-postprecision.json',
                source / 'gpu8-actual-drain.json', source / 'round-1-fresh.json',
                artifact / 'official-after-precision-v4-v1/after_accuracy_only.py',
                artifact / 'official-after-precision-v4-v1/compare_after_accuracy.py',
                artifact / 'official-after-precision-v4-v1/test_original_40_normal_package.py',
                artifact / 'official-after-precision-v4-v1/original-test-adaptation.json',
                artifact / 'input_forensics.py', artifact / 'cooldown.py']
    required.extend(directory / name for name in ['prepare_grant.py', 'run_accuracy_v2.sh', 'launch_accuracy_v2.sh', 'check_cpu_provider.py'])
    original = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014')
    required.extend(original / name for name in ['tests/test_mxfp4_moe_w4a4.py', 'tests/conftest.py', 'tests/mxfp4_utils.py', 'benchmark/accuracy_reference.py'])
    for record in json.loads((source / 'round-1-fresh.json').read_text()):
        required.append(Path(record['output_file'].replace('/root/', '/home/syk/', 1)))
    hashes = {str(path): digest(path) for path in required}
    grant_path = directory / 'gpu11-official-after-accuracy-grant-v2.json'
    if options.verify:
        payload = json.loads(grant_path.read_text())
        assert payload['bound_files_sha256'] == hashes and payload['release'] and payload['gpu'] == 11
        assert not payload['performance_authorized'] and time.time() >= payload['start_epoch']
    else:
        assert not grant_path.exists()
        assert int((root / 'simple-source-compiler-accuracy-gpu11-v2-supervisor.exit').read_text()) == 0
        absent = [3043288, 3043358]
        for pid in absent:
            assert not Path(f'/proc/{pid}').exists(), f'Previous GPU11 owner still alive: {pid}'
        with Path('/tmp/intel-gpu-11.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked_epoch = time.time()
            fcntl.flock(lock, fcntl.LOCK_UN)
        payload = {'authorization': 'latest explicit user: independent W4 genuine normal-v4 AFTER accuracy on GPU11; not performance wave',
                   'release': True, 'gpu': 11, 'cpu_affinity': list(range(64, 80)), 'grant_epoch': time.time(),
                   'start_epoch': time.time() + 10, 'scope': 'B8/ISL4096/OSL16/TP1/EP1',
                   'production_shapes': 5, 'AFTER_rounds': 4, 'original_tests': 40,
                   'independent_FP64_scope': 'max_rows4/expert/fullcolumns; not every production row oracle',
                   'performance_authorized': False, 'timing_events_graphs_warmup': False,
                   'previous_owner': {'handle': 46576, 'CPU_auditor': 7835, 'actual_supervisor_exit': 0,
                                      'PIDs_absent': absent, 'lock11_free_probe': True, 'checked_epoch': checked_epoch,
                                      'results_sha256': digest(previous)},
                   'official_manifest_sha256': package['manifest_sha256'], 'mapped_native_hashes': package['mapped_native_hashes'],
                   'source_ref': '54216962c83b2d489702fd38c58d71116f3a6969',
                   'source_dependency': 'e87870bb7aacf83bfb3be9da685f148316b8df23',
                   'official_dependency': '7b56b936b87f799f80dde60c2fd8d11c1128cebb',
                   'CPU_provider': json.loads((directory / 'CPU-provider.json').read_text()), 'bound_files_sha256': hashes}
        grant_path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'grant': str(grant_path), 'sha256': digest(grant_path), 'start_epoch': payload['start_epoch'], 'gpu': 11}))


if __name__ == '__main__':
    main()

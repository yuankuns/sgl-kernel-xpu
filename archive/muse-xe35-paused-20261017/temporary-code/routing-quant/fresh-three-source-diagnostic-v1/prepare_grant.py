import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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
    original = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    dependency = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    stage = artifact / 'fresh-source-icpx-native-v2'
    for name in ['normal-source-precheck', 'CPU-original-fixtures', 'CPU-preflight']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    assert int((stage / 'cpu-native-acceptance.exit').read_text()) == 0
    assert subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip() == '4cb5e929f84ebcebf98ec6702a26f1ee93190828'
    assert not subprocess.check_output(['git', '-C', str(original), 'status', '--porcelain'], text=True)
    pre = json.loads((stage / 'source-preverify.json').read_text())
    assert subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip() == pre['dependency_HEAD']
    assert pre['dependency_HEAD'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23'
    for root, inventory in [(original, pre['original_tracked']), (dependency, pre['dependency_tracked'])]:
        for name, checksum in inventory.items():
            assert digest(root / name) == checksum, name
    required = [directory / name for name in ['prepare_cpu_fixtures.py', 'accuracy_capture.py', 'run_accuracy.sh',
                                             'launch_accuracy.sh', 'prepare_grant.py', 'normal-source-precheck.json',
                                             'CPU-original-fixtures.json', 'CPU-driver-provenance.json']]
    required += [stage / name for name in ['cpu-native-acceptance.json', 'source-native-package-manifest.json',
                                         'source-preverify.json', 'source-postverify.json']]
    required += [artifact / 'historical-source-failure-cpu-v1/terminal-classification.json',
                 artifact / 'historical-source-failure-cpu-v1/historical-complete.log',
                 artifact / 'source-before-provenance-v3-icpx.json', artifact / 'check_source_precision_package.py',
                 artifact.parent / 'mxfp4-w4a4/input_forensics.py', artifact.parent / 'mxfp4-w4a4/cooldown.py',
                 original / 'tests/test_per_token_group_quant_mxfp4.py', original / 'tests/conftest.py',
                 original / 'src/sycl/TripleOps.cpp', original / 'src/sycl/per_token_group_quant_fp4.cpp']
    for fixture in json.loads((directory / 'CPU-original-fixtures.json').read_text())['fixtures']:
        required.append(directory / fixture['fixture_file'])
    hashes = {str(path): digest(path) for path in required}
    check = json.loads((directory / 'normal-source-precheck.json').read_text())
    driver = json.loads((directory / 'CPU-driver-provenance.json').read_text())
    assert not check['GPU_initialized'] and not driver['GPU_initialized']
    grant_file = directory / 'main-fresh-three-accuracy-gpu8-grant-v1.json'
    if options.verify:
        grant = json.loads(grant_file.read_text())
        assert grant['bound_files_sha256'] == hashes and grant['issuer'] == 'main' and grant['release']
        assert grant['gpu'] == 8 and not grant['performance_authorized'] and not grant['fresh_full_1300_authorized']
        assert time.time() >= grant['start_epoch']
    else:
        assert not grant_file.exists()
        terminal = json.loads((artifact / 'historical-source-failure-cpu-v1/terminal-classification.json').read_text())
        assert terminal['actual_exits']['host-launch'] == 1 and terminal['gpu8_lock_nonblocking_probe_free']
        assert all(not Path(f'/proc/{pid}').exists() for pid in terminal['known_owner_and_postwait_PIDs_absent'])
        epoch = time.time()
        triple = 'sgl_kernel/libsgl-ops-sycl-TripleOps.so'
        manifest = json.loads((stage / 'source-native-package-manifest.json').read_text())
        assert manifest['current_install_files'][triple] == manifest['original_package_files'][triple]
        grant = {'issuer': 'main', 'authorization': 'latest user explicitly delegates main issuer future+120 bounded fresh SOURCE three-nodeid accuracy GPU8',
                 'release': True, 'grant_epoch': epoch, 'start_epoch': epoch + 120, 'gpu': 8, 'CPU_affinity': list(range(64, 80)),
                 'scope': 'B8/ISL4096/OSL16/TP1/EP1; original surface1 is generic tensor edge, not serving override',
                 'source_HEAD': pre['source_HEAD'], 'source_dependency_HEAD': pre['dependency_HEAD'],
                 'normal_SOURCE_site': check['site'], 'actually_mapped_native_sha256': check['mapped_native_sha256'],
                 'CPU_fixtures': json.loads((directory / 'CPU-original-fixtures.json').read_text()), 'driver_provenance': driver,
                 'bound_files_sha256': hashes, 'TripleOps_original_and_fresh_common_sha256': manifest['current_install_files'][triple],
                 'compiler_match_limit': 'only the listed three native freshly icpx compiled; original shared TripleOps/registration/other native reused, not whole SOURCE icpx',
                 'independent_four_components_atol': 0, 'independent_four_components_rtol': 0,
                 'run_original_three_pytest_after_capture_even_if_zero_gate_fails': True,
                 'expected_failure_is_not_success_only_gate': True, 'no_monkeypatch': True,
                 'performance_authorized': False, 'AFTER_authorized': False, 'fresh_full_1300_authorized': False,
                 'compile_authorized': False, 'shared_TripleOps_math_change_authorized': False}
        grant_file.write_text(json.dumps(grant, indent=2) + '\n')
    print(json.dumps({'grant': str(grant_file), 'SHA256': digest(grant_file), 'start_epoch': grant['start_epoch']}))


if __name__ == '__main__':
    main()

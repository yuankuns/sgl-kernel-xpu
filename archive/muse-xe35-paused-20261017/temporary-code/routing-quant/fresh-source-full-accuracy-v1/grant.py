import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 72))
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    dependency = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    stage = artifact / 'fresh-source-icpx-native-v2'
    pre = json.loads((stage / 'source-preverify.json').read_text())
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == pre['source_HEAD']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    assert subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip() == pre['dependency_HEAD']
    for root, inventory in [(source, pre['original_tracked']), (dependency, pre['dependency_tracked'])]:
        for name, expected in inventory.items():
            assert digest(root / name) == expected, name
    for name in ['normal-source-precheck', 'CPU-collect-original', 'CPU-driver-provenance']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    assert int((stage / 'cpu-native-acceptance.exit').read_text()) == 0
    post = artifact / 'fresh-three-source-post-v1'
    for name in ['package-post-fresh', 'CPU-full-fixture-audit', 'CPU-terminal-classification', 'CPU-activation-classification']:
        assert int((post / f'{name}.exit').read_text()) == 0
    classification = json.loads((post / 'GPU8-actual-drain-and-classification.json').read_text())
    assert classification['actual_exits']['original-three-pytest'] == 1
    assert classification['gpu8_lock_probe_free'] and not classification['original_accuracy_gate_passed']
    assert all(not Path(f'/proc/{pid}').exists() for pid in classification['host_owner_PIDs_absent'])
    collect = (directory / 'CPU-collect-original.log').read_text()
    assert '1300 tests collected' in collect
    required = [directory / name for name in ['case_record.py', 'cpu_driver.py', 'grant.py', 'run_gpu.sh', 'launch.sh',
                'normal-source-precheck.json', 'CPU-driver-provenance.json', 'CPU-collect-original.log']]
    required += [post / name for name in ['GPU8-actual-drain-and-classification.json', 'full-fixture-audit.json', 'activation-boundary-classification.json']]
    required += [stage / name for name in ['source-preverify.json', 'source-postverify.json', 'source-native-package-manifest.json', 'cpu-native-acceptance.json']]
    required += [artifact / name for name in ['check_source_precision_package.py', 'cooldown.py', 'source-before-provenance-v3-icpx.json']]
    required += [source / 'tests' / name for name in ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4.py', 'test_topk_sigmoid.py', 'test_routing_benchmark_scope.py', 'conftest.py']]
    hashes = {str(path): digest(path) for path in required}
    check = json.loads((directory / 'normal-source-precheck.json').read_text())
    assert not check['GPU_initialized']
    grant_path = directory / 'main-fresh-full-1300-gpu8-grant-v1.json'
    if args.verify:
        grant = json.loads(grant_path.read_text())
        assert grant['bound_files_sha256'] == hashes and grant['issuer'] == 'main' and grant['release']
        assert grant['CPU_affinity'] == list(range(64, 72)) and grant['gpu'] == 8
        assert time.time() >= grant['start_epoch']
    else:
        assert not grant_path.exists()
        with Path('/tmp/intel-gpu-8.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        epoch = time.time()
        grant = {'issuer': 'main', 'authorization': 'latest explicit user grants fresh SOURCE original full1300 accuracy8 after CPU/source hashes/drain gates0',
                 'release': True, 'grant_epoch': epoch, 'start_epoch': epoch + 120, 'gpu': 8,
                 'CPU_affinity': list(range(64, 72)), 'OMP_NUM_THREADS': 8,
                 'bound_files_sha256': hashes, 'normal_SOURCE_site': check['site'],
                 'actually_mapped_native_sha256': check['mapped_native_sha256'],
                 'source_HEAD': pre['source_HEAD'], 'dependency_HEAD': pre['dependency_HEAD'],
                 'source_current_all_tracked_hashes_verified': True,
                 'previous_owner_actual_drain': classification,
                 'original_skip_reason': 'topk must not exceed the expert count; topk4/8 experts2 BF16/FP32 four original skips',
                 'scope': 'B8/ISL4096/OSL16/TP1/EP1; all original1300, no subset/filter/math/assertion changes',
                 'expected_baseline_failures_remain_true_exit': True, 'performance_authorized': False,
                 'AFTER_authorized': False, 'compile_authorized': False,
                 'shared_TripleOps_recompiled': False, 'shared_TripleOps_SHA256': classification['before_fresh_shared_TripleOps_SHA256']}
        grant_path.write_text(json.dumps(grant, indent=2) + '\n')
        (directory / 'CPU-source-provenance-gate.exit').write_text('0\n')
    print(json.dumps(grant, indent=2))


if __name__ == '__main__':
    main()

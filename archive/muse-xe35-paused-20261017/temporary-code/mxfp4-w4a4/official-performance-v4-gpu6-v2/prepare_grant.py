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
    options = parser.parse_args()
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    assert sorted(os.sched_getaffinity(0)) == list(range(32, 48))
    for name in ['CPU-maintenance-stack', 'CPU-normal-source', 'CPU-normal-after', 'CPU-driver-provider', 'CPU-five-shape-precision-fixtures', 'CPU-preflight']:
        assert int((directory / f'{name}.exit').read_text()) == 0
    current = json.loads((directory / 'CPU-normal-after.json').read_text())
    assert current['current_native_count'] == len(current['mapped_native_hashes']) == 713
    assert current['manifest_sha256'] == '6bb13f86e5736f1ebd3d50b59ee67c032ef44511abef71608e212c991de880ba'
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014')
    dependency = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    stage = artifact / 'fresh-source-icpx-native-v2'
    pre = json.loads((stage / 'source-preverify.json').read_text())
    for root, head, inventory in [(source, pre['source_HEAD'], pre['original_tracked']), (dependency, pre['dependency_HEAD'], pre['dependency_tracked'])]:
        assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == head
        for name, expected in inventory.items():
            assert digest(root / name) == expected, name
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    qk = artifact.parent / 'qk-norm-rope/normal-v4-model-config-perf-v1/drain-check.json'
    qk_drain = json.loads(qk.read_text())
    assert qk_drain['drained'] and not qk_drain['matching_live_own_processes']
    accepted_stack = json.loads((directory / 'CPU-maintenance-stack.json').read_text())
    assert accepted_stack['external_count'] == 0
    assert accepted_stack['stack']['running_kernel'] == '7.0.0-38-generic'
    assert accepted_stack['stack']['xe_srcversion'] == '858285FCD284B7EBB0E410F'
    fm_audit_path = artifact.parent / 'fmha/COORDINATOR_FULL_ICPX_AUDIT_v5.json'
    quiet_path = artifact.parent / 'MAINTENANCE_QUIET_AFTER_FM_v3.json'
    fm_audit = json.loads(fm_audit_path.read_text())
    quiet = json.loads(quiet_path.read_text())
    assert fm_audit['actual_exit'] == fm_audit['default_all_actual_exit'] == fm_audit['install_actual_exit'] == 0
    assert fm_audit['installed_and_python_files_rehashed'] == 1694 and fm_audit['native_binary_inventory_rehashed'] == 735
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    assert all(not sample['active_processes'] for sample in quiet['samples'])
    assert quiet['loaded_stack']['running_kernel'] == accepted_stack['stack']['running_kernel']
    assert quiet['loaded_stack']['loaded_modules']['xe']['srcversion'] == accepted_stack['stack']['xe_srcversion']
    required = [path for path in directory.glob('*') if path.is_file() and path.suffix in ['.py', '.sh']]
    required += [directory / name for name in ['CPU-normal-source.json', 'CPU-normal-after.json', 'CPU-driver-provider.json', 'CPU-five-shape-precision-fixtures.json', 'CPU-maintenance-stack.json']]
    required += [artifact / name for name in ['canonical-source-freeze-v4-icpx.json', 'canonical-v4-icpx/source-binary-manifest.json', 'canonical-v4-icpx/compiler-policy-verified.json', 'COORDINATOR_FULL_ICPX_AUDIT_v4.json', 'input_forensics.py']]
    required += [stage / name for name in ['source-preverify.json', 'source-postverify.json', 'source-native-package-manifest.json', 'cpu-native-acceptance.json', 'compiler-policy-verified.json', 'sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.expanded.commands', 'sgl-ops-sycl-mxfp4_moe_scale_layout.expanded.commands']]
    required += [source / 'benchmark/accuracy_reference.py', qk,
                 artifact / 'official-after-precision-v4-gpu11-v2/source-after-paired-precision.json',
                 artifact / 'official-after-precision-v4-gpu11-v2/original-40-after.log',
                 artifact / 'historical-fresh-source-precision-v3/round-1-fresh.json',
                 artifact / 'historical-fresh-source-precision-v3/original-40-fresh.log']
    records = json.loads((artifact / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    required += [Path(record['output_file'].replace('/root/', '/home/syk/', 1)) for record in records]
    required += [fm_audit_path, quiet_path, artifact / 'official-performance-v4-gpu6-v1/PERFORMANCE_STAGE_RETIRED_v1.json']
    hashes = {str(path): digest(path) for path in required}
    grant_path = directory / 'main-performance-gpu6-grant-v2.json'
    if options.verify:
        grant = json.loads(grant_path.read_text())
        assert grant['bound_files_sha256'] == hashes and grant['issuer'] == 'main' and grant['release']
        assert grant['gpu'] == 6 and grant['CPU_affinity'] == list(range(32, 48))
        assert time.time() >= grant['start_epoch']
    else:
        assert not grant_path.exists()
        with Path('/tmp/intel-gpu-6.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        epoch = time.time()
        grant = {'issuer': 'main', 'authorization': 'latest explicit main user authorizes immutable performance-v2 after FM v5 full25201 and audit1694/735 actual0/compiler-drain0 plus quiet3 61.335s; fresh future+120; CPU32–47 OMP8',
                 'release': True, 'module': 'mxfp4-w4a4', 'version': 'canonical-v4-icpx', 'gpu': 6,
                 'grant_epoch': epoch, 'start_epoch': epoch + 120, 'CPU_affinity': list(range(32, 48)), 'OMP_NUM_THREADS': 8,
                 'performance_authorized': True, 'compile_authorized': False, 'external_compilers_drained': True,
                 'running_stack_verified': True, 'external_and_loaded_stack_snapshot': accepted_stack,
                 'FM_v5_actual_full_audit': fm_audit, 'main_maintenance_quiet_v3': quiet, 'FM_compiler_drain_verified_by_current_process_scan': accepted_stack['external_count'] == 0, 'qk_actual_drain': qk_drain, 'bound_files_sha256': hashes,
                 'normal_current_manifest_SHA256': current['manifest_sha256'], 'normal_mapped_native_SHA256': current['mapped_native_hashes'],
                 'source_head': pre['source_HEAD'], 'source_dependency_head': pre['dependency_HEAD'],
                 'shared_expected_input_scale_raw_SHA256': {record['shape']: record['inputs'] for record in records},
                 'shared_inputs_materialization_scope': 'first SOURCE phase saves complete actual inputs/scales before warmup/timing, verifies old accepted rawSHA, then every phase reuses those PTs and fileSHAs',
                 'all_valid_output_source_hashes': {record['shape']: record['full_valid_output'] for record in records},
                 'scope': 'B8/ISL4096/OSL16/TP1/EP1, original5shapes including257/slack excluded',
                 'BAAB_cycles': 3, 'B': 'genuine fresh SOURCE normal package', 'A': 'genuine current official normal AFTER',
                 'warmup_graph_replays': 5, 'calls_per_graph': 8, 'measured_repeats': 31,
                 'independent_FP64_scope': 'original max_rows4/expert/fullcolumns; not every production row',
                 'maintenance_reappearance_policy': 'last preGPU check and readonly2s watcher; hold125 without touching maintenance; naturally finish current shape, no next shape/phase',
                 'regression_policy': '>5%, robust stable positive or >5% drift hold5, preserve every raw sample/MAD/drift; no ungranted automatic retest/math edits'}
        grant_path.write_text(json.dumps(grant, indent=2) + '\n')
    print(json.dumps({'grant_epoch': grant['grant_epoch'], 'start_epoch': grant['start_epoch'], 'grant_SHA256': digest(grant_path), 'gpu': 6, 'CPU_affinity': grant['CPU_affinity']}))


if __name__ == '__main__':
    main()

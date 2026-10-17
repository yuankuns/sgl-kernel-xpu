import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / 'routing-quant/fresh-current-source-native-v3-preparation'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    DIRECTORY.mkdir(exist_ok=True)
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    old_stage = ROOT / 'routing-quant/fresh-source-icpx-native-v2'
    expected = json.loads((old_stage / 'source-preverify.json').read_text())
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == expected['source_HEAD']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    for name, fingerprint in expected['original_tracked'].items():
        assert digest(source / name) == fingerprint, name
    dependency = Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    for name, fingerprint in expected['dependency_tracked'].items():
        assert digest(dependency / name) == fingerprint, name
    assert subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip() == expected['dependency_HEAD']
    assert not subprocess.check_output(['git', '-C', str(dependency), 'status', '--porcelain'], text=True)
    mapping = {'muse_spark_moe_routing': 'TopKSigMoid', 'topk_sigmoid': 'TopKSigMoid',
               'sgl_per_token_group_quant_fp4': 'per_token_group_quant_fp4',
               'sgl_padded_gather_per_token_group_quant_fp4': 'per_token_group_quant_fp4',
               'silu_and_mul': 'TripleOps', 'mxfp4_moe_scale_to_mn_major': 'mxfp4_moe_scale_layout'}
    test_files = ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4.py', 'test_topk_sigmoid.py',
                  'test_routing_benchmark_scope.py', 'test_silu_and_mul_opmath.py']
    discovered = set()
    for name in test_files:
        tree = ast.parse((source / 'tests' / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'sgl_kernel':
                discovered.update(item.name for item in node.names)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'sgl_kernel':
                discovered.add(node.attr)
    assert discovered <= mapping.keys(), sorted(discovered - mapping.keys())
    (DIRECTORY / 'four-native-coverage.json').write_text(json.dumps({'source_HEAD': expected['source_HEAD'],
        'test_files_SHA256': {name: digest(source / 'tests' / name) for name in test_files},
        'original_test_exposed_kernel_APIs': sorted(discovered), 'op_to_native': mapping,
        'targets': ['sgl-ops-sycl-' + name for name in ['TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout', 'TripleOps']],
        'fifth_tested_math_helper_found': False, 'registration_and_Device_arch_provider_scope': 'normal original unchanged infrastructure, not newly rebuilt math helper',
        'new_GPU_tests_authorized': False}, indent=2) + '\n')
    settings = next((source / 'build/src/CMakeFiles/sgl-ops-sycl-TripleOps.dir').rglob('*.Release.cmake'))
    settings_text = settings.read_text()
    fields = {name: re.search(r'^set\(' + name + r' (.*)\)(?: #.*)?$', settings_text, re.MULTILINE).group(1)
              for name in ('SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_compile_definitions')}
    lines = (source / 'build/build.ninja').read_text().splitlines()
    device = next(lines[index + 1] for index, line in enumerate(lines) if line.startswith('build ') and
                  'sgl-ops-sycl-TripleOps' in line and 'sycl_device_obj.o' in line and ': CUSTOM_COMMAND' in line)
    reference = {'fields': fields, 'actual_device_link_command': device, 'reference_file': str(settings),
                 'reference_file_SHA256': digest(settings), 'reference_ninja_SHA256': digest(source / 'build/build.ninja'),
                 'not_legacy_binary_compile_provenance': True,
                 'TripleOps_original_GRF': 'auto-large-GRF unchanged; do not invent explicit512 for this generic original target',
                 'resource_only_link_parallel_change': 'original256 to approved inner1'}
    (DIRECTORY / 'TripleOps-original-generated-flags.json').write_text(json.dumps(reference, indent=2) + '\n')
    (DIRECTORY / 'TripleOps-original.Release.cmake').write_bytes(settings.read_bytes())
    previous = {str(path.relative_to(old_stage)): digest(path) for path in old_stage.rglob('*') if path.is_file()}
    assert int((old_stage / 'supervisor.exit').read_text()) == int((old_stage / 'native-build.exit').read_text()) == 0
    (DIRECTORY / 'old-stage-protected-SHA256.json').write_text(json.dumps(previous, indent=2) + '\n')
    w4 = ROOT / 'mxfp4-w4a4/official-performance-v4-gpu6-v2'
    assert int((w4 / 'host-launch.exit').read_text()) == int((w4 / 'gpu-runtime.exit').read_text()) == int((w4 / 'performance-analysis.exit').read_text()) == 5
    for phase, role in enumerate(['source', 'after', 'after', 'source'] * 3):
        assert int((w4 / f'phase-{phase}-{role}.exit').read_text()) == 0
    bf = ROOT / 'bf16-gemm/formal-v4-cleanconfirm-gpu4-v1'
    assert int((bf / 'launcher.exit').read_text()) == int((bf / 'GPU-round.exit').read_text()) == 0
    simple = ROOT / 'simple-model-verified-performance-v2-gpu7-run1'
    simple_phase = simple / 'round-0-slot-0-before'
    assert int((simple_phase / 'wrapper.exit').read_text()) == int((simple_phase / 'memory-semantic.exit').read_text()) == 1
    owners = [int((w4 / name).read_text()) for name in ['host-waiter.pid', 'host-supervisor.pid', 'maintenance-monitor.pid']]
    owners += [json.loads((bf / name).read_text())['pid'] for name in ['supervisor.started.json', 'GPU-round.started.json']]
    owners += [json.loads((simple / 'launch-plan.json').read_text())['host_pid'], json.loads((simple_phase / 'wrapper-owned.json').read_text())['PID']]
    assert all(not Path('/proc', str(pid)).exists() for pid in owners), owners
    for gpu in [4, 6, 7]:
        with Path(f'/tmp/intel-gpu-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    spec = importlib.util.spec_from_file_location('maintenance', w4 / 'maintenance_snapshot.py')
    maintenance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maintenance)
    snapshot = maintenance.snapshot()
    assert snapshot['external_count'] == 0, snapshot
    (DIRECTORY / 'maintenance-and-stable-owner-drain.json').write_text(json.dumps({'snapshot': snapshot,
        'known_host_owner_PIDs_absent': owners, 'physical4_6_7_locks_free': True,
        'BF_clean_actual_exit': 0, 'W4_performance_actual_exit': 5, 'GPU7_v2_actual_exit': 1,
        'external_processes_stopped_or_modified': False}, indent=2) + '\n')
    report = {'epoch': time.time(), 'CPU_preparation_actual_exit': 0, 'source_HEAD': expected['source_HEAD'],
        'source_TripleOps_SHA256': digest(source / 'src/sycl/TripleOps.cpp'), 'dependency_HEAD': expected['dependency_HEAD'],
        'original_tracked_source_and_dependency_hashes_verified': True, 'old_stage_file_count': len(previous),
        'old_stage_untouched': True, 'coverage_four_native': True, 'fifth_tested_helper_found': False,
        'CPU_affinity': sorted(os.sched_getaffinity(0)), 'outer_jobs': 2, 'inner_jobs': 1,
        'file_compiler_subtree_budget_GiB': 8, 'tree_budget_GiB': 16, 'GPU_work_started': False}
    (DIRECTORY / 'CPU-preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

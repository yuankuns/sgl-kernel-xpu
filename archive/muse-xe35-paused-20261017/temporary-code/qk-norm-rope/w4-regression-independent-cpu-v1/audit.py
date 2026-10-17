import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parents[1]
W4 = ARTIFACT / 'mxfp4-w4a4'
PERF = W4 / 'official-performance-v4-gpu6-v2'
FRESH = W4 / 'fresh-source-icpx-native-v2'
SOURCE = FRESH / 'source'
AFTER = ARTIFACT.parents[0] / 'pr/muse-mxfp4-w4a4-xe35'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalize(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    text = re.sub(r'^\s*#include[^\n]*', '', text, flags=re.M)
    return re.sub(r'\s+', '', text.replace('_xe35', ''))


def generated_fields(path):
    names = ('SYCL_executable', 'SYCL_host_compiler', 'SYCL_compile_flags', 'SYCL_compile_definitions', 'CMAKE_HOST_FLAGS')
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r'set\((\w+) (.*?)\)\s*(?:#.*)?$', line)
        if match and match[1] in names:
            result[match[1]] = match[2]
    assert set(result) == set(names)
    return {'path': str(path), 'sha256': digest(path), 'fields': result}


def main():
    assert sorted(os.sched_getaffinity(0)) == list(range(16)) and 'torch' not in sys.modules
    grant = read(PERF / 'main-performance-gpu6-grant-v2.json')
    for name in ('graph_phase.py', 'analyze.py', 'cpu_provider.py', 'run_gpu.sh'):
        path = PERF / name
        assert digest(path) == grant['bound_files_sha256'][str(path)]
    stored = read(PERF / 'performance-analysis.json')
    order = ['source', 'after', 'after', 'source'] * 3
    phases = []
    evidence = {}
    for phase, role in enumerate(order):
        prefix = PERF / f'phase-{phase}-{role}'
        assert prefix.with_suffix('.exit').read_text().strip() == '0'
        records = read(prefix.with_suffix('.json'))
        assert len(records) == 5
        phases.append(records)
        for suffix in ('.json', '.log', '.exit'):
            path = prefix.with_suffix(suffix)
            evidence[str(path)] = digest(path)
    recomputed = []
    for case_index, original in enumerate(stored['shapes']):
        rows = [phase[case_index] for phase in phases]
        for phase, row in enumerate(rows):
            assert row['kind'] == order[phase] and row['phase'] == phase and row['shape'] == original['shape']
            assert row['input_hashes'] == rows[0]['input_hashes'] and row['full_valid_output_hash'] == rows[0]['full_valid_output_hash']
            assert row['genuine_replay_NaN_sentinel_full_valid_output_proof']
            assert row['calls_per_graph'] == 8 and row['measured_repeats'] == 31 and row['warmup_graph_replays'] == 5
            assert len(row['raw_per_call_ms']) == len(row['raw_graph_ms']) == 31
            assert all(math.isfinite(value) and value > 0 for value in row['raw_graph_ms'])
            assert all(math.isclose(graph / 8, call, rel_tol=1e-12) for graph, call in zip(row['raw_graph_ms'], row['raw_per_call_ms']))
            median = statistics.median(row['raw_per_call_ms'])
            assert median == row['median_ms']
            assert statistics.median(abs(value - median) for value in row['raw_per_call_ms']) == row['MAD_ms']
        samples = {role: [value for row in rows if row['kind'] == role for value in row['raw_per_call_ms']] for role in ('source', 'after')}
        result = {'shape': original['shape'], 'full_raw_phase_samples_ms': [row['raw_per_call_ms'] for row in rows]}
        for role, values in samples.items():
            median = statistics.median(values)
            result[f'{role}_median_ms'] = median
            result[f'{role}_MAD_ms'] = statistics.median(abs(value - median) for value in values)
            assert result[f'{role}_median_ms'] == original[f'{role}_median_ms']
            assert result[f'{role}_MAD_ms'] == original[f'{role}_MAD_ms']
        result['change_pct'] = (result['after_median_ms'] / result['source_median_ms'] - 1) * 100
        assert result['change_pct'] == original['change_pct']
        cycles = []
        drift = []
        for cycle in range(3):
            quartet = rows[cycle * 4:cycle * 4 + 4]
            cycles.append((statistics.median(quartet[1]['raw_per_call_ms'] + quartet[2]['raw_per_call_ms']) / statistics.median(quartet[0]['raw_per_call_ms'] + quartet[3]['raw_per_call_ms']) - 1) * 100)
            drift.append({'cycle': cycle, 'source_BB_drift_pct': (quartet[3]['median_ms'] / quartet[0]['median_ms'] - 1) * 100,
                'after_AA_drift_pct': (quartet[2]['median_ms'] / quartet[1]['median_ms'] - 1) * 100})
        assert cycles == original['three_BAAB_cycle_changes_pct'] and drift == original['within_cycle_drift']
        result.update({'three_BAAB_cycle_changes_pct': cycles, 'within_cycle_drift': drift,
            'source_BB_max_abs_drift_pct': max(abs(item['source_BB_drift_pct']) for item in drift),
            'after_AA_max_abs_drift_pct': max(abs(item['after_AA_drift_pct']) for item in drift)})
        recomputed.append(result)
    assert not stored['performance_gate_passed']
    terminal = {name: int((PERF / f'{name}.exit').read_text()) for name in ('performance-analysis', 'gpu-runtime', 'host-launch')}
    assert terminal == {'performance-analysis': 5, 'gpu-runtime': 5, 'host-launch': 5}
    native = read(FRESH / 'cpu-native-acceptance.json')
    original = read(FRESH / 'source-preverify.json')
    manifest = read(W4 / 'canonical-v4-icpx/source-binary-manifest.json')
    assert native['actual_exit'] == 0 and native['source_revision'] == original['source_HEAD'] == '54216962c83b2d489702fd38c58d71116f3a6969'
    assert native['dependency_pin'] == original['dependency_HEAD'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23'
    assert manifest['cutlass_pin'] == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
    assert digest(W4 / 'canonical-v4-icpx/source-binary-manifest.json') == grant['normal_current_manifest_SHA256']
    mappings = [('src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp', 'src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp'),
        ('src/sycl/kernels/moe/xe35/device/grouped_gemm_runner.hpp', 'src/sycl/xe35/moe/device/grouped_gemm_runner.hpp'),
        ('src/sycl/kernels/moe/xe35/collective/moe_dtype_policy.hpp', 'src/sycl/xe35/moe/collective/moe_dtype_policy.hpp')]
    mechanical = []
    for source_name, after_name in mappings:
        assert digest(SOURCE / source_name) == original['original_tracked'][source_name]
        assert digest(AFTER / after_name) == manifest['source']['files'][after_name]['sha256']
        assert normalize((SOURCE / source_name).read_text()) == normalize((AFTER / after_name).read_text())
        mechanical.append({'source_file': str(SOURCE / source_name), 'source_sha256': digest(SOURCE / source_name),
            'after_file': str(AFTER / after_name), 'after_sha256': digest(AFTER / after_name),
            'CPP_body_equal_after_only_comments_includes_whitespace_and_xe35_identifier_tag_removal': True})
    precision_files = ['src/sycl/xe35/moe/collective/moe_bf16_reorder.hpp', 'src/sycl/xe35/moe/collective/moe_array_epilogue.hpp']
    for name in precision_files:
        assert digest(AFTER / name) == manifest['source']['files'][name]['sha256']
        evidence[str(AFTER / name)] = digest(AFTER / name)
    fields = {}
    for role, build in [('source', FRESH / 'build'), ('after', ARTIFACT / 'canonical-icpx-pool-v4/build')]:
        path = next((build / 'src/CMakeFiles/sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.dir').glob('**/*.Release.cmake'))
        fields[role] = generated_fields(path)
        for flag in ('-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-fp64-conv-emu', '-fsycl-targets=spir64_gen'):
            assert flag in fields[role]['fields']['SYCL_compile_flags']
    source_policy = read(FRESH / 'compiler-policy-verified.json')
    target_policy = next(item for item in source_policy['targets'] if item['target'] == 'sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4')
    assert fields['source']['sha256'] == target_policy['generated_settings_sha256']
    for path in [PERF / 'graph_phase.py', PERF / 'analyze.py', PERF / 'main-performance-gpu6-grant-v2.json', PERF / 'performance-analysis.json',
        PERF / 'CPU-normal-source.json', PERF / 'CPU-normal-after.json', FRESH / 'compiler-policy-verified.json',
        FRESH / 'cpu-native-acceptance.json', FRESH / 'source-preverify.json', W4 / 'canonical-v4-icpx/compiler-policy-verified.json']:
        evidence[str(path)] = digest(path)
    providers = {}
    for role, name, map_key in [('source', 'CPU-normal-source.json', 'mapped_native_sha256'), ('after', 'CPU-normal-after.json', 'mapped_native_hashes')]:
        data = read(PERF / name)
        providers[role] = {'package': data['package'], 'mapped_native_count': len(data[map_key]),
            'selected_native_sha256': {str(Path(path).name): checksum for path, checksum in data[map_key].items()
                if any(token in Path(path).name for token in ('common_ops', 'moe_grouped_mm_mxfp4_w4a4', 'mxfp4_moe_scale_layout'))}}
    evidence[str(ROOT / 'REPORT.md')] = digest(ROOT / 'REPORT.md')
    result = {'independent_CPU_audit_actual_exit': 0, 'CPU_affinity': list(range(16)), 'raw_phase_count': 12,
        'all_five_shapes': recomputed, 'device_event_graph_sample_count': 1860, 'original_terminal_exits': terminal,
        'original_performance_gate_passed': False, 'providers': providers, 'source_ref': native['source_revision'],
        'source_dependency_pin': native['dependency_pin'], 'after_dependency_pin': manifest['cutlass_pin'],
        'mechanical_comparisons': mechanical, 'generated_actual_settings': fields, 'input_evidence_sha256': evidence,
        'normalization_does_NOT_prove_dependency_or_generated_device_code_equivalence': True,
        'shared_canonical_snapshot_now_does_NOT_retain_W4_CPP': not (ARTIFACT / 'canonical-icpx-pool-v4/source/src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp').exists(),
        'after_math_source_provenance_verified_using_frozen_target_SHA_vs_original_package_manifest': True,
        'raw_PT_activations_outputs_and_all_native_binaries_NOT_rehashed_this_light_review': True,
        'GPU_compile_Torch_ISA_not_executed': True, 'root_cause_NOT_proven': True,
        'no_existing_PR_READY_driver_raw_math_tolerance_or_Mill_build_outputs_modified': True, 'GPU6_QK_grant': False}
    (ROOT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Independent CPU audit actual0: 12 phases/1860 graph samples/statistics verified; original terminal5 retained; dispatch bodies agree; differences do not establish root cause')
    for item in recomputed[-2:]:
        print(item['shape'], 'change_pct=', item['change_pct'], 'SOURCE_BB_max_abs_pct=', item['source_BB_max_abs_drift_pct'])


if __name__ == '__main__':
    main()

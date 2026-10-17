import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ARTIFACT = RUN.parent
PERF = ARTIFACT / 'normal-v4-model-config-perf-v1'
sys.path.insert(0, str(PERF))
from run_perf import digest, local_path, verify


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def link_command(ninja, target):
    selected = False
    with ninja.open() as stream:
        for line in stream:
            if line.startswith('build '):
                selected = line.startswith(f'build src/CMakeFiles/{target}.dir/') and '_sycl_device_obj.o: CUSTOM_COMMAND' in line
            elif selected and line.startswith('  COMMAND = '):
                return line.strip()[len('COMMAND = '):]
    raise AssertionError(f'Missing existing device-link command {target}')


def main():
    assert set(os.sched_getaffinity(0)) == set(range(16))
    data = verify()
    for name in ('metadata-comparison', 'normal-host-dtype-source', 'normal-host-dtype-port'):
        assert (RUN / f'{name}.exit').read_text().strip() == '0'
    comparison = read(RUN / 'metadata-comparison.json')
    assert comparison['all_executable_prefixes_through_last_EOT_byte_identical'] and comparison['kernel_pairs'] == 80
    dtype_source = read(RUN / 'normal-host-dtype-source.json')
    dtype_port = read(RUN / 'normal-host-dtype-port.json')
    assert dtype_source['cases'] == dtype_port['cases'] and not dtype_source['XPU_initialized'] and not dtype_port['XPU_initialized']
    required = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-fp64-conv-emu']
    source_policy_path = local_path(data['source_provider']['source_native_compiler_proof'])
    source_policy = read(source_policy_path)
    port_policy_path = ARTIFACT / 'canonical-v4-icpx/compiler-policy-verified.json'
    port_policy = read(port_policy_path)
    source_recipe = local_path(source_policy['compile_recipe'])
    port_recipe = ARTIFACT.parent / 'canonical-icpx-pool-v4/build/src/CMakeFiles/sgl-ops-sycl-FusedQKNormRope_xe35.dir/sycl/xe35/sgl-ops-sycl-FusedQKNormRope_xe35_gen_FusedQKNormRope.cpp.o.Release.cmake'
    recipes = {}
    for role, path in (('source', source_recipe), ('port', port_recipe)):
        text = path.read_text()
        lines = [line for line in text.splitlines() if line.startswith(('set(SYCL_compile_flags ', 'set(SYCL_host_compiler ',
            'set(SYCL_compile_definitions ', 'set(CMAKE_HOST_FLAGS ', 'set(SYCL_executable '))]
        assert lines and all(flag in text for flag in required)
        recipes[role] = {'path': str(path), 'sha256': digest(path), 'precision_flags_present': required,
            'relevant_raw_configuration_lines': lines, 'SYCL_INTEL_TARGET35_present': 'SYCL_INTEL_TARGET=35' in text}
    for role, policy in (('source', source_policy), ('port', port_policy)):
        assert policy['compiler_id'] == 'IntelLLVM'
        assert policy['cxx'] == policy['sycl'] == policy['host'] == '/opt/gfx-deps/dpcpp/bin/icpx'
        assert policy['c'] == '/opt/gfx-deps/dpcpp/bin/icx'
    port_link = link_command(ARTIFACT / 'canonical-v4-icpx/build.ninja', 'sgl-ops-sycl-FusedQKNormRope_xe35')
    source_link = source_policy['device_link_recipe']
    for command in (source_link, port_link):
        assert '-device' in command and 'cri' in command and '-cl-fp32-correctly-rounded-divide-sqrt' in command
        assert '-fsycl-max-parallel-link-jobs=1' in command and '-fsycl-targets=spir64_gen' in command
    flags = {'SOURCE_compiler_policy_SHA': digest(source_policy_path), 'PORT_compiler_policy_SHA': digest(port_policy_path),
        'actual_cxx_SYCL_host': '/opt/gfx-deps/dpcpp/bin/icpx', 'actual_C': '/opt/gfx-deps/dpcpp/bin/icx',
        'recipes': recipes, 'SOURCE_existing_AOT_link_command': source_link, 'PORT_existing_AOT_link_command': port_link,
        'new_compiler_command_executed': False,
        'host_flags_are_NOT_identical': True,
        'PORT_host_additions_relative_to_SOURCE': ['-O3', '-fvisibility=hidden', '-fvisibility-inlines-hidden'],
        'SOURCE_original_profiling_defines_not_in_PORT': ['CUTLASS_SYCL_PROFILING_ENABLED', 'COMPAT_PROFILING_ENABLED'],
        'shared_SYCL_submission_helper_difference': 'SOURCE void q.submit(cgf); PORT returns event via pass-through track wrapper, profiling branch disabled by PORT flags',
        'host_differences_cost_NOT_measured_or_inferred': True}
    save(RUN / 'flags-host-path-evidence.json', flags)
    framework = Path('/home/syk/sglang/python/sglang')
    files = [framework / 'kernels/ops/attention/fused_qknorm_rope.py', framework / 'kernels/ops/attention/position.py',
        framework / 'srt/model_executor/forward_batch_info.py', framework / 'srt/model_executor/runner_utils/buffers.py',
        framework / 'srt/hardware_backend/xpu/graph_runner/xpu_graph_runner.py', framework / 'srt/managers/schedule_batch.py',
        ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35/src/sycl/FusedQKNormRope.cpp',
        ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35/src/sycl/xe35/FusedQKNormRope.cpp',
        ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35/src/sycl/xe35/FusedQKNormRopeFrequency.h',
        ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35/src/sycl/xe35/FusedQKNormRopeKernel.h',
        ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35/src/sycl/SYCLHelpers.h',
        ARTIFACT / 'native-icpx-cpu-v3/optimized-source/source/src/sycl/SYCLHelpers.h']
    source_files = {str(path): digest(path) for path in files}
    fixture = [{key: case[key] for key in ('name', 'group', 'case', 'fixture', 'parameters', 'expected_SOURCE_output', 'expected_SOURCE_output_sha256')} for case in data['cases']]
    save(RUN / 'shared-19-fixture-plan.json', fixture)
    plan = {'version': 'qk-controlled-method-diagnostic-plan-v1', 'CPU_only_preparation': True,
        'current_CPU_affinity': '0-15', 'current_GPU6_owner': 'main handoff Mill W4', 'GPU_authorized': False,
        'future_device_requires_new_explicit_main_grant': [4, 6], 'future_runtime_CPU_requires_new_main_assignment': True,
        'scope': data['scope'], 'fixtures': {'count': 19, 'actual_config_semantic_synthetic': 17, 'legacy_representatives': 2,
            'sha256': digest(RUN / 'shared-19-fixture-plan.json'), 'all_shared_original_inputs_SHA': {case['name']: case['fixture']['sha256'] for case in data['cases']},
            'no_fixture_case_selection_or_input_math_dtype_changes': True},
        'providers': {'SOURCE_commit': data['source_provider']['source_commit'],
            'SOURCE_manifest_SHA': data['source_provider']['source_manifest_sha256'], 'SOURCE_native_SHA': data['source_provider']['source_qk_library_sha256'],
            'SOURCE_original_package_native_rebuild_scope': data['source_provider']['source_native_build_scope'],
            'PORT_normal_full_manifest_SHA': data['canonical_manifest_sha256'], 'PORT_native_SHA': data['port_native_SHA'],
            'compiler_policy_SHA': data['compiler_policy_sha256'], 'PR_source_digest': data['source_digest']},
        'framework_and_source_file_SHA': source_files, 'flags_evidence_SHA': digest(RUN / 'flags-host-path-evidence.json'),
        'existing_normal_CPU_probe_SHA': {role: digest(RUN / f'normal-host-dtype-{role}.json') for role in ('source', 'port')},
        'metadata_comparison_SHA': digest(RUN / 'metadata-comparison.json'),
        'retained_frozen_previous_analyzer_SHA': data['analysis_driver_sha256'],
        'retained_previous_full_performance_analysis_SHA': digest(PERF / 'analysis-final.json'),
        'preserve_previous_regression_confirmed_plus7_3491_and_raw_drift40_8245': True,
        'methods': {
            'per_call_device_events': {'reset': 'full original QKVO copy before every call outside event',
                'timed': 'start.record -> actual unchanged adapter(position cast if int64) -> generic native enqueue -> end.record',
                'meaning': 'GPU timestamps enclosing device operations and possibly CPU submission starvation; not pure native kernel time'},
            'captured_reset_plus_adapter': {'capture_body': '32 times full QKVO reset then SAME genuine framework adapter call; original positions dtype unchanged',
                'timed': 'one XPUGraph replay enclosed by device events /32',
                'includes': ['full QKVO reset GPU copy', 'int64-to-int32 GPU cast for 17 semantic fixtures', 'native GPU work', 'graph scheduling/replay effects'],
                'host_native_dispatch_predicate': 'run during actual capture after genuine warmup, then captured GPU work replayed; not whole-model raw replay'},
            'captured_reset_only': {'capture_body': '32 times exact same full QKVO original reset on same destination; no adapter/no norm/no RoPE',
                'timed': 'one XPUGraph replay enclosed by device events /32', 'postcheck': 'full output bits equal ORIGINAL input'}},
        'difference_diagnostic_only': {'raw': 'index-aligned sample captured_reset_plus_adapter minus captured_reset_only; keep negatives',
            'summary': 'difference of method medians plus MAD of raw index-aligned deltas',
            'not_additive_reason': 'reset/cast/cache/frequency/graph dependencies and replay/submission overhead differ; subtraction cannot identify pure production kernel latency',
            'not_correctness_matched_preopt_speedup': True},
        'fixed_protocol': {'BAAB_order': ['port', 'source', 'source', 'port'], 'rounds': 3, 'every_phase_all_cases': 19,
            'samples': 21, 'calls_per_sample': 32, 'warmup_per_method': 5,
            'method_order_within_each_case': 'all three methods each sample; deterministic three-way rotation by round/slot/case/sample, same rule both providers',
            'case_order': 'all19 canonical forward even rounds/reversed odd; no reduced-case retests',
            'wrapper_lock_and_initial_cooling_seconds': 5, 'synchronize_then_case_cooling_seconds': 2,
            'no_extra_auto_rounds_or_tuning': True},
        'correctness': {'dtype': 'BF16', 'atol': 0.005, 'rtol': 0.005, 'full_CPU_reference': 'same immutable reviewed reference',
            'full_QKV_SOURCE_PORT_uint16_comparison': True, 'tail_V16_plus_O_gate128_bitwise': True,
            'reset_each_logical_call': True, 'source_and_port_every_method_full_input_SHA_identical': True,
            'all_phase_postcheck_with_first_phase_full_output_witness_each_provider_method_case': True,
            'capture_or_validation_failure': 'real exit1 and traceback; no fake op/shim/manual load/native bridge/math fix/fallback timing'},
        'future_release_required_fields': ['stable physical GPU4-or6 exact lease', 'new future epoch+120', 'fresh compiler+package-manager drain JSON SHA',
            'current normal full package/compiler/PR source/SOURCE manifest/native SHA', 'all19 fixture SHA',
            'actual future runner/analyzer/launcher SHA', 'normal SOURCE155/PORT712 native maps SHA',
            'same loaded Torch/libtorch_xpu/libsycl/UR/LevelZero driver paths+SHA before/after', 'device arch3.5 after XPU available check',
            'independent runtime CPU assignment', 'graph capture and reference/output full-bit validation evidence'],
        'future_runner_not_implemented_or_executed': True, 'future_runner_SHA': None,
        'core_loaded_stack_process_maps_NOT_collected_by_existing_SGL_only_maps': True,
        'old_graph_timing_not_reused_for_speedup': True, 'no_GPU_compile_package_manager_activity': True}
    save(RUN / 'method-plan.json', plan)
    fields = {'plan_SHA': digest(RUN / 'method-plan.json'), 'fixture_plan_SHA': digest(RUN / 'shared-19-fixture-plan.json'),
        'flags_evidence_SHA': digest(RUN / 'flags-host-path-evidence.json'), 'metadata_comparison_SHA': digest(RUN / 'metadata-comparison.json'),
        'normal_HOST_dtypes_SOURCE_SHA': digest(RUN / 'normal-host-dtype-source.json'),
        'normal_HOST_dtypes_PORT_SHA': digest(RUN / 'normal-host-dtype-port.json'),
        'GPU_authorized': False, 'CPU_affinity': sorted(os.sched_getaffinity(0))}
    save(RUN / 'plan-fields-hashes.json', fields)
    print(json.dumps(fields, indent=2))


if __name__ == '__main__':
    main()

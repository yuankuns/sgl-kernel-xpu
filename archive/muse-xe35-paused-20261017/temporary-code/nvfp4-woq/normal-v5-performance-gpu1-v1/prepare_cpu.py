import argparse
import json
import os
import re
from pathlib import Path

import policy


def prepare(attempt):
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    old = json.loads((policy.ACCURACY / 'AFTER_ACCURACY_READY.json').read_text())
    audit = json.loads((policy.ACCURACY / 'independent-completed-CPU-audit-v1.json').read_text())
    current_provenance = policy.original_provenance()
    assert current_provenance == audit['provenance'], 'Current body/helpers/native flags/source freeze differs from completed audit'
    assert audit['actual_independent_CPU_audit_exit'] == 0
    assert old['host_supervisor_actual_exit'] == old['GPU_wrapper_actual_exit'] == old['postcheck_CPU_actual_exit'] == 0
    assert old['tail_records'] == old['production_records'] == 16 and old['all_valid_outputs_and_prep_c_sf_scales_bitwise']
    assert old['all_known_owned_descendants_PID_startticks_drained']
    for name, sha in old['raw_sha256'].items():
        assert policy.digest(policy.ACCURACY / name) == sha, 'Original accuracy evidence seal changed'
    source_policy = json.loads((policy.FRESH / 'compiler-policy-verified.json').read_text())
    assert source_policy['original_source_flags_preserved'] and source_policy['NV_original_GRF512_preserved']
    original_targets = ('nvfp4_woq', 'blockwise_moe_nvfp4_woq', 'mxfp4_moe_scale_layout', 'blockwise_moe_mxfp8')
    assert set(source_policy['targets']) == {'sgl-ops-sycl-' + name for name in original_targets}
    source_target = source_policy['targets']['sgl-ops-sycl-blockwise_moe_nvfp4_woq']
    source_AOT = source_target['actual_AOT_command']
    after_AOT = [line for line in (policy.AFTER / 'build.ninja').read_text().splitlines() if line.strip().startswith('COMMAND =') and 'sgl-ops-sycl-blockwise_moe_nvfp4_woq_sycl_device_obj.o' in line and '/icpx ' in line]
    assert len(after_AOT) == 1
    for command in (source_AOT, after_AOT[0]):
        assert '-cl-intel-512-GRF-per-thread' in command and '-fsycl-max-parallel-link-jobs=1' in command
    source_flags = re.search(r'^set\(SYCL_compile_flags (.*?)\) # list$', source_target['actual_compile_metadata'], re.MULTILINE).group(1).split(';')
    after_policy = json.loads((policy.AFTER / 'compiler-policy-verified.json').read_text())
    assert all(flag in source_flags for flag in after_policy['required_flags'])
    providers = {}
    for label in ('source', 'after'):
        assert (policy.DIRECTORY / f'CPU-provider-{label}-{attempt}.exit').read_text().strip() == '0'
        record = json.loads((policy.DIRECTORY / f'CPU-provider-{label}-{attempt}.json').read_text())
        assert not record['GPU_initialized'] and record['after_manifest_sha256'] == policy.EXPECTED_MANIFEST
        providers[label] = policy.digest(policy.DIRECTORY / f'CPU-provider-{label}-{attempt}.json')
    policy.save(policy.DIRECTORY / 'CPU_READY.json', {'actual_exit': 0, 'GPU_started': False, 'compiler_started': False, 'CPU_affinity': list(range(80, 96)), 'manifest_sha256': policy.EXPECTED_MANIFEST, 'source_digest': policy.EXPECTED_SOURCE, 'provider_SHA256': providers, 'provider_files': {label: f'CPU-provider-{label}-{attempt}.json' for label in ('source', 'after')}, 'current_original_provenance': current_provenance, 'dependencies_SHA256': policy.dependencies(), 'SOURCE_tested_native_targets': list(original_targets), 'SOURCE_registration_limit': 'preserved historical GNU common_ops/all other native libraries; four fresh icpx native targets only, not full SOURCE/allregistration icpx', 'SOURCE_cutlass': 'e87870bb7aacf83bfb3be9da685f148316b8df23', 'AFTER_cutlass': '7b56b936b87f799f80dde60c2fd8d11c1128cebb', 'original50tests_skips': 0, 'original50tests': 50, 'tails': 16, 'production': 16, 'old_full_valid_outputs_c_sf_bits': True, 'production_reference': audit['reference_scope'], 'source_SYCL_compile_flags': source_flags, 'AFTER_required_precision_flags': after_policy['required_flags'], 'SOURCE_AOT_command': source_AOT, 'AFTER_AOT_command': after_AOT[0], 'GRF512_both': True, 'flag_difference_limit': 'same preserved precision/spir64_gen/CRI local512 strategy; SOURCE e878 finite tested-native harness versus official7b56 normal default-all feature/registration graph, different macros/includes/helper adaptations; not identical whole toolchain graph', 'GRF_history': 'GPU9 regression triggered investigation; authoritative d983 already512, restored omission only; no GPU9-selected algorithm/dispatch/precision tuning', 'production_shapes': policy.PRODUCTION, 'representative_shapes_total_routed_rows': policy.REPRESENTATIVE, 'representative_factory': 'original bench_nvfp4_woq_group_gemm.build_inputs unchanged quantizers; four default NVWOQ MXFP8 cases; plain_mxfp8 upper-bound is not port performance', 'formal_grant_created': False, 'performance_accepted': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=('source', 'after'))
    parser.add_argument('--seal-ready', action='store_true')
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'r[0-9]+', args.attempt)
    policy.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    if args.provider:
        record = policy.original_provider(args.provider)
        policy.save(policy.DIRECTORY / f'CPU-provider-{args.provider}-{args.attempt}.json', record)
        print(json.dumps({'actual_exit': 0, 'provider': args.provider, 'XPU_initialized': record['GPU_initialized']}))
    elif args.seal_ready:
        prepare(args.attempt)
        print(json.dumps({'actual_exit': 0, 'GPU_started': False, 'compiler_started': False, 'formal_grant_created': False}))
    else:
        parser.error('CPU preparation action required')


if __name__ == '__main__':
    main()

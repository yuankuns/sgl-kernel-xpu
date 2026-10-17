import json
import os
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
MIB = 1024 ** 2
GIB = 1024 ** 3
REPRESENTATIVE = ((2, 128, 128, 256), (2, 128, 256, 512),
                  (4, 256, 256, 512), (2, 512, 512, 512))
PRODUCTION = ((256, 64, 8192, 8192), (256, 262144, 8192, 8192),
              (256, 64, 8192, 4096), (256, 262144, 8192, 4096))


def representative_bytes(experts, rows, channels, inner):
    return {'a_q_fp8': rows * inner, 'a_s_uint8': rows * inner // 32,
            'b_fp4_uint8': experts * channels * inner // 2,
            'b_sf_fp8': experts * channels * inner // 16,
            'b_gs_fp32': experts * 4,
            'b_mx_fp8_factory_upper_bound': experts * channels * inner,
            'b_smx_uint8_factory_upper_bound': experts * channels * inner // 32,
            'problem_sizes_int32': experts * 3 * 4, 'offsets_int32': experts * 4,
            'c_bf16': experts * channels * inner // 16 * 2,
            'sfmx_uint8': experts * channels * inner // 32,
            'actual_output_fp32': rows * channels * 4,
            'expected_output_fp32_full_rows': rows * channels * 4,
            'counts_int64_upper_bound': experts * 8}


def main():
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    coverage = json.loads((DIRECTORY / 'PRODUCTION_SHAPE_COVERAGE.json').read_text())
    assert coverage['actual_exit'] == 0
    representatives = []
    total = 0
    for shape in REPRESENTATIVE:
        amounts = representative_bytes(*shape)
        payload = sum(amounts.values())
        total += payload * 4
        representatives.append({'E_totalR_N_K': shape, 'bytes_per_phase': amounts,
                                'payload_bytes_four_phases': payload * 4})
    assert total < 64 * MIB
    production = []
    for experts, rows, channels, inner in PRODUCTION:
        amounts = {'output_fp32': rows * channels * 4,
                   'c_bf16': experts * channels * inner // 16 * 2,
                   'sfmx_uint8': experts * channels * inner // 32}
        production.append({'E_totalR_N_K': [experts, rows, channels, inner],
                           'first_SOURCE_cache_bytes': amounts})
    production_total = sum(sum(case['first_SOURCE_cache_bytes'].values()) for case in production)
    assert production_total == int(23.50390625 * GIB)
    result = {'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'GPU_started': False, 'compiler_started': False, 'new_grant_created': False,
              'production_shape_scope_expanded': False,
              'representative_accuracy_only_proposal': {
                  'cases': representatives, 'phases': ['SOURCE', 'AFTER', 'AFTER', 'SOURCE'],
                  'payload_bytes_all_four_phases': total,
                  'tensor_serialization_and_manifest_ceiling_bytes': 64 * MIB,
                  'provider_logs_trace_ceiling_bytes': 64 * MIB,
                  'temporary_and_misc_ceiling_bytes': 128 * MIB,
                  'total_new_artifact_ceiling_bytes': 256 * MIB,
                  'additional_free_space_headroom_bytes': GIB,
                  'total_admission_reservation_bytes': GIB + 256 * MIB,
                  'package_or_native_copy_bytes': 0,
                  'save_contiguous_clones_required': True,
                  'unquantized_factory_local_weights_not_saved_or_claimed': True,
                  'future_main_accuracy_and_disk_grant_required': True},
              'production_future_performance_cache_only_not_admitted': {
                  'cases': production, 'one_baseline_cache_payload_bytes': production_total,
                  'one_baseline_cache_payload_GiB': production_total / GIB,
                  'not_complete_performance_disk_budget': True,
                  'raw_samples_provider_logs_temporary_margin_not_in_this_payload': True},
              'note': 'Existing 16 production records cover all four shapes. No new large production accuracy job, no allocation, no performance grant.'}
    with (DIRECTORY / 'DISK_BUDGET_CPU.json').open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'representative_four_phase_payload_bytes': total,
                      'future_accuracy_disk_reservation_GiB': 1.25,
                      'production_first_cache_payload_GiB': production_total / GIB,
                      'GPU_started': False}))


if __name__ == '__main__':
    main()

import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
sys.path.insert(0, str(ART / 'launch-v3-icpx'))
from matched_baseline_provenance import package_preflight, provenance


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    precision = ART / 'v5-model51-gpu10-preparation'
    raw_path = precision / 'normal-v5-gpu10-accuracy.json'
    summary_path = precision / 'normal-v5-gpu10-summary.json'
    audit_path = precision / 'normal-v5-89-full-output-bitwise-CPU-audit.json'
    raw_sha = digest(raw_path)
    assert raw_sha == '90fbd175fbb525799e77df8e943714c7d5f3c4b20ad9c3d8b4c4c54cd19d1643'
    raw, summary, audit = [json.loads(path.read_text()) for path in (raw_path, summary_path, audit_path)]
    assert raw_sha == summary['raw_sha256'] == audit['raw_sha256']
    assert raw['all89_fullrefs_and_source_wholebitwise_PASS'] and summary['actual_exit'] == audit['runtime_actual_exit'] == 0
    assert audit['whole_saved_output_pairs_checked'] == 89 and audit['BF16_bitwise_mismatches'] == 0
    assert audit['whole_output_elements'] == 1884094464
    model = [case for case in raw['cases'] if case['family'] == 'model51']
    representative = [case for case in raw['cases'] if case['family'] == 'regression38']
    assert len(model) == 51 and len(representative) == 38
    assert sum(case['checked_elements'] for case in model) == 808255488
    assert sum(case['checked_elements'] for case in representative) == 1075838976
    assert sum(case['name'].startswith('model-prefill') for case in model) == summary['model51_actual_private_prefill_routes'] == 6
    assert sum(case['name'].startswith('model-decode') for case in model) == summary['model51_actual_private_decode_routes'] == 45
    outputs = {}
    for case in raw['cases']:
        assert case['accuracy'] == 'PASS' and case['reference_PASS'] and case['full_output_bitwise']
        assert case['actual_metadata_private_route_PASS'] and case['source_port_max_abs'] == 0
        assert case['atol'] == case['rtol'] == 0.005 and case['coverage'] == 'full'
        path = precision / 'normal-v5-full-outputs' / f'{case["family"]}-{case["name"]}.pt'
        outputs[str(path)] = digest(path)
        assert outputs[str(path)] == case['saved_full_output_file_sha256']
    for name in ('model51-normal-v5-gpu10-accuracy-v1', 'model51-normal-v5-gpu10-barrier-v1',
                 'model51-normal-v5-89-full-output-CPU-audit-v1', 'model51-normal-v5-grant-close-CPU-v1'):
        assert (ART / f'{name}.exit').read_text().strip() == '0'
    closed = json.loads((ART / 'canonical-v5-icpx/gpu10-precision-only-barrier-v1.json').read_text())
    assert closed['closed'] and not closed['coordinator_release'] and not closed['runtime_reuse_permitted']
    assert summary['GPU10_runtime_drained_lock_released'] and not summary['live_GPU10_runtime_processes']
    seal = {'epoch': time.time(), 'CPU_only': True, 'independent_raw_SHA_matches_summary_and_CPU_audit': True,
        'raw_sha256': raw_sha, 'summary_sha256': digest(summary_path), 'independent_bitwise_audit_sha256': digest(audit_path),
        'normal_manifest_sha256': raw['canonical_manifest_sha256'], 'current_source_digest': raw['source_digest'],
        'model51': {'cases': 51, 'checked_elements': 808255488, 'private_prefill': 6, 'private_decode': 45,
                    'fullrefs_and_SOURCE_wholebits': 51},
        'representative38': {'cases': 38, 'checked_elements': 1075838976, 'fullrefs_and_SOURCE_wholebits': 38},
        'total_checked_elements': 1884094464, 'all89_full_output_file_SHA_rehashed': outputs,
        'actual_runtime_summary_drain_audit_grant_close_exit': 0, 'GPU10_grant_closed': True,
        'not_performance_or_end_to_end_model_acceptance': True}
    (ROOT / 'GPU10_V5_PRECISION_FINAL_SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
    reports = {}
    for role in ('source', 'preopt'):
        package, hashes = package_preflight(role)
        native = provenance(role)
        label = 'fresh-optimized' if role == 'source' else 'fresh-preopt'
        precision_path = ART / f'source-only-v1-{label}.json'
        precision_raw = json.loads(precision_path.read_text())
        assert (ART / f'source-only-v1-{label}.exit').read_text().strip() == '0'
        assert precision_raw['all_38_reference_PASS'] and len(precision_raw['cases']) == 38
        kernels = native['native_kernels']
        assert len(kernels) == 3
        for record in kernels.values():
            assert record['library_sha256'] == hashes[record['library']]
            assert record['sycl_host_compiler'] == '/opt/gfx-deps/dpcpp/bin/icpx'
        reports[role] = {'actual_current_package': str(package), 'current_package_file_count': len(hashes),
            'actual_original_native_provenance': native,
            'genuine_three_tested_native_icpx_current_SHA_verified': True,
            'complete_original_package_all_icpx': False,
            'remaining_registration_and_non_target_components': 'historical GNU/common_ops and other components; not a substitute for fresh tested math native',
            'original38_precision_actual_exit': 0, 'original38_fullrefs': 38,
            'original38_precision_raw_sha256': digest(precision_path),
            'original38_checked_elements': sum(case['checked_elements'] for case in precision_raw['cases']),
            'actual_model51_preopt_accuracy_proof': False if role == 'preopt' else True,
            'preopt51_required_before_any_preopt51_timing': role == 'preopt',
            'missing_tested_native_compile_required': False,
            'if_whole_package_icpx_is_required': 'not available; requires separate authorized genuine compilation, cannot label mixed package all-icpx',
            'no_compile_GPU_started': True}
    (ROOT / 'ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps({'finalseal': 0, 'raw_sha256': raw_sha, 'model51_private_prefill': 6,
        'model51_private_decode': 45, 'total_elements': 1884094464,
        'current_preopt_three_actual_icpx_native': True, 'preopt38_fullrefs': 38,
        'preopt51_accuracy': 'pending new grant', 'all_package_icpx_claim': False,
        'compile_GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()

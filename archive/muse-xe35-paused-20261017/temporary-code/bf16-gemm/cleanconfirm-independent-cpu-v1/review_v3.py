import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent
ORIGINAL = ROOT / 'formal-v4-cleanconfirm-gpu4-v1'
VALIDATED = ROOT / 'source-native-icpx-v1-accuracy'
PREFIX = 'canonical-v4-icpx-cleanconfirm-gpu4-v1-BAAB'
ROLES = ('source', 'port', 'port', 'source')
PATHS = ('native_kernel', 'packed_kernel', 'native_api', 'packed_api')
COLUMNS = (8192, 18432, 202048)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def host_path(path):
    if str(ROOT).startswith('/root/'):
        return Path(str(path).replace('/home/syk/', '/root/', 1))
    return Path(str(path).replace('/root/', '/home/syk/', 1))


def tensor_digest(tensor):
    checksum = hashlib.sha256()
    for chunk in tensor.split(1024):
        checksum.update(memoryview(chunk.contiguous().view(torch.uint8).numpy()))
    return checksum.hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main():
    assert os.sched_getaffinity(0) == set(range(80, 96))
    assert os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu'
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    seal = read(ORIGINAL / 'EVIDENCE_MANIFEST_FINAL.json')
    assert seal['actual_exit'] == 0 and seal['performance_accepted'] is False
    checked_files = {}
    for name, expected in seal['files_sha256'].items():
        assert digest(ORIGINAL / name) == expected, 'Original sealed raw changed: ' + name
        checked_files[name] = expected
    sidecar = ROOT / 'cleanconfirm-cross-source-postcheck-cpu-v1'
    for name, expected in seal['CPU_sidecar_files_sha256'].items():
        assert digest(sidecar / name) == expected
    manifest = read(ORIGINAL / 'manifest.json')
    early_manifest_differences = []
    for name, expected in manifest['files_sha256'].items():
        actual = digest(ORIGINAL / name)
        if actual != expected:
            assert name == 'supervisor.log' and seal['files_sha256'][name] == actual
            checksum = hashlib.sha256()
            prefix_bytes = 0
            match = None
            for line in (ORIGINAL / name).read_bytes().splitlines(keepends=True):
                checksum.update(line)
                prefix_bytes += len(line)
                if checksum.hexdigest() == expected:
                    match = prefix_bytes
            assert match is not None and match < (ORIGINAL / name).stat().st_size
            early_manifest_differences.append({'file': name, 'early_SHA256': expected, 'final_SHA256': actual, 'early_exact_prefix_bytes': match, 'final_bytes': (ORIGINAL / name).stat().st_size, 'classification': 'Exact old sealed log prefix retained; final manifest captures append-only terminal lines. Old manifest/raw unchanged.'})
    grant = read(ORIGINAL / 'fresh-grant.json')
    assert grant['gpu'] == 4 and grant['CPU_affinity'] == '0-15'
    assert grant['calls'] == 20 and grant['repeats'] == 20 and grant['warmup'] == 5
    assert grant['minimum_BAAB_rounds'] == 1 and not grant['all_eight_cards_parallel_verified']
    close(grant['epoch'] - grant['issued_epoch'], 120)
    frozen = read(ORIGINAL / 'frozen-implementation-manifest.json')
    for name, expected in frozen['files_sha256'].items():
        assert digest(ROOT / frozen['version'] / name) == expected
    for name in ('supervisor.exit', 'GPU-round.exit', 'analyze.exit', 'CPU-protocol.exit', 'CPU-source-provider.exit', 'cpu-gate.exit', 'CPU-units.exit'):
        assert (ORIGINAL / name).read_text().strip() == '0'
    assert (ORIGINAL / 'CPU-cross-source-postcheck.exit').read_text().strip() == '1'
    assert 'libpti_view.so.0' in (ORIGINAL / 'CPU-cross-source-postcheck.log').read_text()
    assert (sidecar / 'CPU-cross-source-postcheck-v2.exit').read_text().strip() == '0'
    summary = read(ORIGINAL / (PREFIX + '-summary.json'))
    assert summary['order'] == list(ROLES) and len(summary['rows']) == 12
    assert summary['prefill'] is None and summary['performance_accepted'] is False
    assert summary['coordinator_review_pending'] and not summary['all_eight_cards_parallel_verified']
    runs = [read(ORIGINAL / f'{PREFIX}-{index}-{role}.json') for index, role in enumerate(ROLES)]
    source = read(VALIDATED / 'fresh.json')
    source_summary = read(VALIDATED / 'summary.json')
    assert source_summary['independent_full_CPU_references_passed'] and source_summary['accuracy_passed']
    assert not source_summary['performance_collected'] and not source_summary['official_AFTER_tested']
    inputs = read(ORIGINAL / 'CPU_PROTOCOL_INPUT_INTERFACE.json')['input_hashes']
    mappings = {}
    events = {}
    for index, (role, run) in enumerate(zip(ROLES, runs)):
        assert run['label'] == role and run['gpu_mask'] == '4'
        assert run['atol'] == 0.005 and run['rtol'] == 0.005
        scope = run['scope']
        assert {key: scope[key] for key in ('serving_batch', 'isl', 'osl', 'tp', 'ep')} == {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
        assert scope['warmup'] == 5 and scope['calls'] == 20 and scope['repeat'] == 20
        assert run['actual_CPU_affinity'] == list(range(16))
        assert run['fresh_grant']['sha256'] == digest(ORIGINAL / 'fresh-grant.json')
        assert run['canonical_manifest_sha256'] == manifest['AFTER_manifest_sha256']
        assert run['icpx_source_proofs']['source_digest'] == manifest['source_digest']
        assert run['tested_SOURCE_compiler_control']['compiler_control_precision_verified']
        assert [case['n'] for case in run['cases']] == list(COLUMNS)
        assert (ORIGINAL / f'{PREFIX}-{index}-{role}.exit').read_text().strip() == '0'
        mapped = {}
        package = host_path(run['package']).resolve().parent
        for name, expected in run['loaded_libraries'].items():
            path = host_path(name).resolve()
            assert path.is_relative_to(package) and digest(path) == expected
            mapped[str(path)] = expected
        assert len(mapped) == (156 if role == 'source' else 712)
        mappings[str(index)] = {'role': role, 'package': str(package), 'static_file_SHA_verified_count': len(mapped)}
        samples = []
        for line in (ORIGINAL / f'{PREFIX}-{index}-{role}.log').read_text().splitlines():
            if line.startswith('{'):
                item = json.loads(line)
                if set(item) == {'n', 'name', 'repeat', 'ms'}:
                    samples.append(item)
        expected_order = [(columns, repeat, name) for columns in COLUMNS for repeat in range(20) for name in (PATHS if repeat % 2 == 0 else PATHS[::-1])]
        assert [(item['n'], item['repeat'], item['name']) for item in samples] == expected_order
        assert all(math.isfinite(item['ms']) and item['ms'] > 0 for item in samples)
        events[str(index)] = samples
    refs = []
    crossrun = []
    cross_source = []
    input_checks = []
    generator = torch.Generator().manual_seed(42)
    for case_index, columns in enumerate(COLUMNS):
        activation = torch.randn((8, 8192), dtype=torch.bfloat16, generator=generator)
        weight = torch.randn((columns, 8192), dtype=torch.bfloat16, generator=generator)
        input_sha = tensor_digest(activation)
        weight_sha = tensor_digest(weight)
        assert input_sha == inputs[case_index]['activation_sha256'] == source['cases'][case_index]['input_sha256']
        assert weight_sha == inputs[case_index]['weight_sha256'] == source['cases'][case_index]['weight_sha256']
        reference = activation.float() @ weight.float().T
        baseline_path = VALIDATED / source['cases'][case_index]['outputs']
        assert digest(baseline_path) == source['cases'][case_index]['outputs_sha256']
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)
        assert set(baseline) == set(PATHS)
        outputs = []
        for index, (role, run) in enumerate(zip(ROLES, runs)):
            case = run['cases'][case_index]
            assert case['m'] == 8 and case['k'] == 8192 and case['dtype'] == 'bf16'
            assert case['input_sha256'] == input_sha and case['weight_sha256'] == weight_sha
            output_path = ORIGINAL / case['outputs']
            assert digest(output_path) == case['outputs_sha256']
            output = torch.load(output_path, map_location='cpu', weights_only=True)
            assert set(output) == set(PATHS)
            outputs.append(output)
            for name in PATHS:
                actual = output[name]
                assert actual.device.type == 'cpu' and actual.dtype == torch.float32 and tuple(actual.shape) == (8, columns)
                assert case['accuracy'][name]['output_dtype'] == 'torch.bfloat16'
                torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.005)
                error = (actual - reference).abs()
                maximum = error.max().item()
                ratio = (error / (0.005 + 0.005 * reference.abs())).max().item()
                close(maximum, case['accuracy'][name]['max_abs'])
                close(ratio, case['accuracy'][name]['max_error_over_tolerance'])
                refs.append({'run_index': index, 'role': role, 'n': columns, 'path': name, 'elements': actual.numel(), 'reference_dtype': 'CPU FP32', 'output_original_dtype': 'BF16', 'saved_output_dtype': 'FP32 promotion of BF16 output', 'atol': 0.005, 'rtol': 0.005, 'max_abs': maximum, 'max_error_over_tolerance': ratio, 'passed': True})
                assert torch.equal(actual.contiguous().view(torch.uint8), baseline[name].contiguous().view(torch.uint8))
                cross_source.append({'run_index': index, 'role': role, 'n': columns, 'path': name, 'elements': actual.numel(), 'equal': True})
                for repeat, sample in enumerate(case['samples_ms'][name]):
                    matches = [item for item in events[str(index)] if item['n'] == columns and item['repeat'] == repeat and item['name'] == name]
                    assert len(matches) == 1 and matches[0]['ms'] == sample
                assert len(case['samples_ms'][name]) == 20
                close(case['medians_ms'][name], statistics.median(case['samples_ms'][name]))
        for index in (1, 2, 3):
            for name in PATHS:
                assert torch.equal(outputs[0][name].contiguous().view(torch.uint8), outputs[index][name].contiguous().view(torch.uint8))
                crossrun.append({'case_index': case_index, 'comparison_run_index': index, 'path': name, 'elements': outputs[0][name].numel(), 'equal': True})
        input_checks.append({'n': columns, 'activation_elements': activation.numel(), 'weight_elements': weight.numel(), 'activation_SHA256': input_sha, 'weight_SHA256': weight_sha, 'all4runs_and_validated_SOURCE_full_bytes_SHA_equal': True})
        del activation, weight, reference, baseline, outputs, output, actual, error
    assert len(refs) == 48 and len(crossrun) == 36 and len(cross_source) == 48
    old_cross = read(ORIGINAL / 'CPU_CROSS_VALIDATED_SOURCE_FULL_BITS.json')
    key = lambda item: (item['run_index'], item['role'], item['n'], item['path'])
    assert len({key(item) for item in cross_source}) == 48
    assert len({key(item) for item in old_cross['complete_checks']}) == 48
    assert sorted(cross_source, key=key) == sorted(old_cross['complete_checks'], key=key)
    assert [{key: value for key, value in item.items() if key != 'comparison_run_index'} for item in crossrun] == summary['full_tensor_bitwise_checks']
    recomputed = []
    for row in summary['rows']:
        columns = int(row['case'].split('-N')[1].split('-')[0])
        case_index = COLUMNS.index(columns)
        name = row['path']
        samples = [run['cases'][case_index]['samples_ms'][name] for run in runs]
        assert row['all_raw_samples_ms'] == samples and row['outliers_removed'] == 0
        medians = [statistics.median(values) for values in samples]
        source_median = statistics.median(samples[0] + samples[3])
        port_median = statistics.median(samples[1] + samples[2])
        values = {'source_ms': source_median, 'port_ms': port_median, 'change_pct': 100 * (port_median / source_median - 1), 'source_drift_pct': 100 * (medians[3] / medians[0] - 1), 'port_drift_pct': 100 * (medians[2] / medians[1] - 1), 'source_MAD_ms': statistics.median(abs(value - source_median) for value in samples[0] + samples[3]), 'port_MAD_ms': statistics.median(abs(value - port_median) for value in samples[1] + samples[2])}
        for key, value in values.items():
            close(value, row[key])
        assert row['run_medians_ms'] == medians
        assert row['run_MAD_ms'] == [statistics.median(abs(value - median) for value in sample) for sample, median in zip(samples, medians)]
        needs_retest = abs(values['change_pct']) > 5 or abs(values['source_drift_pct']) > 5
        stable_positive = values['change_pct'] > 0 and min(medians[1], medians[2]) > max(medians[0], medians[3])
        assert needs_retest == row['needs_retest'] and stable_positive == row['small_stable_positive_regression']
        recomputed.append({'case': row['case'], 'path': name, **values, 'original_analyzer_needs_retest': needs_retest, 'original_analyzer_small_stable_positive': stable_positive, 'positive_absolute_gap_below_both_pooled_MAD': values['change_pct'] <= 0 or abs(port_median - source_median) < min(values['source_MAD_ms'], values['port_MAD_ms'])})
    chronology_path = ROOT / 'BF_NV_V5_TIMELINE_SIDECAR.json'
    assert digest(chronology_path) == grant['old_BF_compiler_chronology_sha256']
    chronology = read(chronology_path)
    for name, expected in chronology['evidence_sha256'].items():
        assert digest(ROOT.parent / name) == expected
    assert chronology['NV_configure_overlap_before_BF_drain_seconds'] > 0
    assert chronology['BF_individual_events_wallclock_timestamps_available'] is False
    assert chronology['NV_configure_subcompiler_argv_recorded'] is False
    assert not torch.xpu.is_initialized()
    report = {'actual_independent_CPU_review_exit': 0, 'review_epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__, 'XPU_initialized': False, 'original_artifact': str(ORIGINAL), 'original_final_manifest_SHA256': digest(ORIGINAL / 'EVIDENCE_MANIFEST_FINAL.json'), 'original_raw_seals_rehashed': len(checked_files), 'early_manifest_append_only_log_difference': early_manifest_differences, 'original_file_seals': checked_files, 'fresh_grant_SHA256': digest(ORIGINAL / 'fresh-grant.json'), 'original_roles': list(ROLES), 'single4process_round_not3rounds': True, 'scope': 'B8 decode M8/K8192, N8192/18432/202048, four native/packed kernel/API paths, original BF16 output and FP32 full CPU oracle; not prefill/model', 'full_CPU_FP32_refs_recomputed': refs, 'full_crossrun_uint8_bits_checks': crossrun, 'full_validated_SOURCE_uint8_bits_checks': cross_source, 'input_weight_fullbytes_seed42_recomputed': input_checks, 'static_provider_mapping_file_SHA_checks': mappings, 'original_raw_device_event_samples': sum(len(values) for values in events.values()), 'statistics_recomputed_without_outlier_drop': recomputed, 'old_configure_overlap_preserved': chronology, 'old_configure_overlap_SHA256': digest(chronology_path), 'old_cross_SOURCE_CPU_exit1_preserved': True, 'old_cross_SOURCE_CPU_exit1_reason': 'Original libpti_view.so.0 LD import omission; independent sidecar-v2 exit0 preserved, not GPU/math failure.', 'SOURCE_provenance_limit': 'Genuine original package/legacy GNU common_ops registration with limited icpx tested-native only; original e878 vs AFTER public7b56 remains, not all-source/all-registration icpx.', 'old_GPU_drain_record_only': read(ORIGINAL / 'GPU-drained.json'), 'current_GPU_lock_or_ownership_tested': False, 'bounded_maintenance_snapshot_not_continuous_quiet_proof': True, 'root_PR_DRAFT_current_SHA_equals_historical_final_seal': digest(ROOT / 'PR_DRAFT.md') == seal['root_PR_DRAFT_sha256'], 'review_not_automatic_performance_acceptance': True, 'performance_accepted': False, 'full_prefill_or_model_or_all8_claimed': False, 'GPU_started': False, 'compiler_started': False, 'PR_source_native_oldraw_modified': False}
    with (DIRECTORY / 'REVIEW.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_independent_CPU_review_exit': 0, 'refs': len(refs), 'crossrun_full_bits': len(crossrun), 'validated_SOURCE_full_bits': len(cross_source), 'full_input_weight_SHA_checks': len(input_checks), 'device_event_raw_samples_reaudited': report['original_raw_device_event_samples'], 'original_sealed_files_rehashed': len(checked_files), 'old_configure_overlap_retained': True, 'XPU_initialized': False, 'performance_accepted': False}, indent=2))


if __name__ == '__main__':
    main()

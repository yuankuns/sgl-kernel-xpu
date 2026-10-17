import json
import os

from checks import ROOT, ROUND, digest, grant_gate, write_json


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('MoE whole tensor analysis is CPU96–111 only')
    import torch

    manifest, manifest_file, site, protected, grant = grant_gate(True)
    for name in ['CPU-collection.exit', 'cpu-gate.exit', 'after.exit', 'pytest.exit', 'GPU-round.exit']:
        if (ROUND / name).read_text().strip() != '0':
            raise RuntimeError('Actual pre-wave gate exit0 required: ' + name)
    run = json.loads((ROUND / 'after.json').read_text())
    source = json.loads((ROOT / 'source-native-icpx-v1-accuracy/fresh.json').read_text())
    assert run['accuracy_passed'] and run['full_tensor_bitwise_passed']
    assert run['gpu_mask'] == '10' and run['actual_container_CPU_affinity'] == list(range(96, 112))
    assert not run['timing_collected'] and not run['graphs_or_events_created'] and not run['warmup_performed']
    assert not run['performance_collected'] and not run['formal_perf_barrier_consumed']
    assert len(run['cases']) == len(source['cases']) == 6
    assert run['prefill']['pairs'] == source['prefill']['pairs'] == 262144
    assert run['prefill']['rejection_verified'] and source['prefill']['rejection_verified']
    shared_inputs = torch.load(ROUND / 'shared-SOURCE-inputs.pt', weights_only=True, map_location='cpu')
    checks = []
    for after, before in zip(run['cases'], source['cases']):
        assert after['case'] == before['case'] and after['dtype'] == before['dtype']
        assert after['input_sha256'] == before['input_sha256']
        assert after['tokens'] == before['tokens'] == 8 and after['topk'] == before['topk'] == 8
        assert after['experts'] == before['experts'] == 256 and after['alignment'] == before['alignment'] == 4
        assert digest(ROUND / after['outputs']) == after['outputs_sha256']
        before_file = ROOT / 'source-native-icpx-v1-accuracy' / before['outputs']
        assert digest(before_file) == before['outputs_sha256']
        actual = torch.load(ROUND / after['outputs'], weights_only=True, map_location='cpu')
        expected = torch.load(before_file, weights_only=True, map_location='cpu')
        assert len(actual) == len(expected) == 4
        for index, (output, reference) in enumerate(zip(actual, expected)):
            assert output.shape == reference.shape and output.dtype == reference.dtype == torch.int32
            identical = torch.equal(output, reference)
            checks.append({'case': after['case'], 'output': index, 'elements': output.numel(), 'equal': identical})
            assert identical, 'Complete normal AFTER versus actual SOURCE tensor mismatch'
    assert len(checks) == 24 and len(shared_inputs) == 6
    pytest_log = (ROUND / 'pytest.log').read_text()
    if '30 passed' not in pytest_log or ' skipped' in pytest_log:
        raise RuntimeError('All original 30 precision tests must actually pass without skips')
    summary = {'version': 'canonical-v4-icpx', 'scope': 'genuine complete MoE AFTER pre-wave accuracy only',
               'actual_normal_import_verified': True, 'all_mapped_sgl_DSO_SHA_verified': True,
               'installed_native_count': len(manifest['binaries']), 'mapped_DSO_count': len(run['loaded_libraries']),
               'independent_CPU_exact_cases_passed': 6, 'atol': 0, 'rtol': 0,
               'full_tensor_bitwise_checks': checks, 'full_tensor_bitwise_passed': True,
               'original_precision_tests_actual_passed': 30, 'CPU_provider_collection_actual_exit': 0,
               'actual_container_CPU_affinity': list(range(96, 112)),
               'AFTER_manifest_sha256': digest(manifest_file), 'SOURCE_native_manifest_sha256': grant['grant']['source_native_manifest_sha256'],
               'fresh_grant_sha256': grant['sha256'], 'driver_sha256': grant['grant']['driver_sha256'],
               'shared_SOURCE_inputs_sha256': grant['grant']['shared_SOURCE_inputs_sha256'],
               'protected_source_post': protected, 'performance_collected': False,
               'timing_events_graphs_warmup_added': False, 'formal_perf_barrier_consumed': False,
               'prefill_fastpath_supported': False, 'unsupported_prefill_contract_rejection_verified': True,
               'GPU_wrapper_already_returned': True, 'performance_acceptance_complete': False}
    write_json(ROUND / 'summary.json', summary)
    print(json.dumps({'accuracy_passed': True, 'full_tensor_bitwise_checks': len(checks),
                      'original_precision_tests_passed': 30, 'performance_collected': False}), flush=True)


if __name__ == '__main__':
    main()

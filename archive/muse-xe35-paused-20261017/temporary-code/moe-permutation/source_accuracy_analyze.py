import json
import os
from pathlib import Path

from source_accuracy_package_check import digest


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
        raise RuntimeError('Full tensor analysis must be CPU only')
    import torch

    root = Path(__file__).resolve().parent
    stage = root / 'source-native-icpx-v1-accuracy'
    historical = json.loads((stage / 'historical.json').read_text())
    fresh = json.loads((stage / 'fresh.json').read_text())
    expected_gpu = '4' if root.name == 'bf16-gemm' else '10'
    for label, run in [('historical', historical), ('fresh', fresh)]:
        assert (stage / (label + '.exit')).read_text().strip() == '0'
        assert run['version'] == 'source-native-icpx-v1' and run['accuracy_only']
        assert run['gpu_mask'] == expected_gpu and not run['timing_collected'] and not run['graphs_or_events_created']
        assert run['all_mapped_sgl_dso_hashes_verified']
        assert run['scope']['serving_batch'] == 8 and run['scope']['isl'] == 4096 and run['scope']['osl'] == 16
        assert run['scope']['tp'] == 1 and run['scope']['ep'] == 1
    assert historical['device_properties'] == fresh['device_properties']
    assert historical['fresh_grant']['sha256'] == fresh['fresh_grant']['sha256']
    assert historical['source_HEAD'] == fresh['source_HEAD']
    assert len(historical['cases']) == len(fresh['cases']) == (3 if root.name == 'bf16-gemm' else 6)
    checks = []
    for before, after in zip(historical['cases'], fresh['cases']):
        assert before['input_sha256'] == after['input_sha256']
        if root.name == 'bf16-gemm':
            assert before['weight_sha256'] == after['weight_sha256']
            assert before['m'] == after['m'] == 8 and before['k'] == after['k'] == 8192
            assert before['n'] == after['n'] and before['dtype'] == after['dtype'] == 'bf16'
        else:
            assert before['case'] == after['case'] and before['dtype'] == after['dtype']
        for record in [before, after]:
            assert digest(stage / record['outputs']) == record['outputs_sha256']
        before_outputs = torch.load(stage / before['outputs'], weights_only=True, map_location='cpu')
        after_outputs = torch.load(stage / after['outputs'], weights_only=True, map_location='cpu')
        if isinstance(before_outputs, dict):
            assert set(before_outputs) == set(after_outputs) == {'native_kernel', 'packed_kernel', 'native_api', 'packed_api'}
            pairs = [(name, before_outputs[name], after_outputs[name]) for name in before_outputs]
        else:
            assert len(before_outputs) == len(after_outputs) == 4
            pairs = [(str(index), before_outputs[index], after_outputs[index]) for index in range(4)]
        for name, first, second in pairs:
            assert first.shape == second.shape and first.dtype == second.dtype
            identical = torch.equal(first, second)
            checks.append({'case': before.get('case', 'M8-N' + str(before.get('n')) + '-K8192'),
                           'output': name, 'elements': first.numel(), 'equal': identical})
            assert identical, 'Full SOURCE compiler-control bitwise mismatch: ' + name
    if root.name == 'moe-permutation':
        assert historical['prefill']['rejection_verified'] and fresh['prefill']['rejection_verified']
        assert historical['prefill']['pairs'] == fresh['prefill']['pairs'] == 262144
    expected_checks = 12 if root.name == 'bf16-gemm' else 24
    assert len(checks) == expected_checks
    summary = {'version': 'source-native-icpx-v1', 'scope': 'SOURCE accuracy only', 'gpu': int(expected_gpu),
               'historical_SOURCE_native_sha256': historical['tested_native_sha256'],
               'fresh_SOURCE_native_sha256': fresh['tested_native_sha256'],
               'source_manifest_sha256': fresh['fresh_grant']['manifest_sha256'],
               'fresh_grant_sha256': fresh['fresh_grant']['sha256'],
               'independent_full_CPU_references_passed': True, 'full_tensor_bitwise_checks': checks,
               'full_tensor_bitwise_passed': True, 'accuracy_passed': True,
               'performance_collected': False, 'official_AFTER_tested': False,
               'baseline_scope': fresh['baseline_scope'], 'GPU_round_drained_before_analysis': True,
               'prefill_fastpath_supported': False if root.name == 'moe-permutation' else None}
    (stage / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'accuracy_passed': True, 'full_bitwise_checks': len(checks), 'performance_collected': False}), flush=True)


if __name__ == '__main__':
    main()

import hashlib
import json
import os
import time
from pathlib import Path


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(16)):
        raise RuntimeError('CPU-only0–15 postcheck required')
    import torch
    folder = Path(__file__).resolve().parent.parent / 'formal-v4-cleanconfirm-gpu4-v1'
    root = folder.parent
    source_folder = root / 'source-native-icpx-v1-accuracy'
    source = json.loads((source_folder / 'fresh.json').read_text())
    inputs = json.loads((folder / 'CPU_PROTOCOL_INPUT_INTERFACE.json').read_text())['input_hashes']
    checks = []
    for index, label in enumerate(['source', 'port', 'port', 'source']):
        run = json.loads((folder / ('canonical-v4-icpx-cleanconfirm-gpu4-v1-BAAB-' + str(index) + '-' + label + '.json')).read_text())
        for case_index, case in enumerate(run['cases']):
            baseline = source['cases'][case_index]
            assert case['input_sha256'] == baseline['input_sha256'] == inputs[case_index]['activation_sha256']
            assert case['weight_sha256'] == baseline['weight_sha256'] == inputs[case_index]['weight_sha256']
            expected_path = source_folder / baseline['outputs']
            actual_path = folder / case['outputs']
            assert hashlib.sha256(expected_path.read_bytes()).hexdigest() == baseline['outputs_sha256']
            assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == case['outputs_sha256']
            expected = torch.load(expected_path, weights_only=True)
            actual = torch.load(actual_path, weights_only=True)
            assert set(expected) == set(actual)
            for name in expected:
                identical = torch.equal(expected[name], actual[name])
                assert identical
                checks.append({'run_index': index, 'role': label, 'n': case['n'], 'path': name, 'elements': actual[name].numel(), 'equal': identical})
    assert len(checks) == 48 and not torch.xpu._initialized
    record = {'epoch': time.time(), 'actual_exit': 0, 'complete_checks': checks, 'complete_check_count': 48,
              'all_full_input_and_weight_bytes_SHA_match_validated_genuine_SOURCE': True,
              'all_new_whole_tensors_bitwise_equal_validated_genuine_SOURCE': True, 'GPU_started': False}
    (folder / 'CPU_CROSS_VALIDATED_SOURCE_FULL_BITS.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'full_checks': 48, 'input_weight_SHA_verified': True, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()

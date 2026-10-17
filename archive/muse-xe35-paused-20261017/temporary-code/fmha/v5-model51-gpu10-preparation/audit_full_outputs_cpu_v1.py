import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    actual_exit = int((ART / 'model51-normal-v5-gpu10-accuracy-v1.exit').read_text().strip())
    raw_path = ROOT / 'normal-v5-gpu10-accuracy.json'
    raw = json.loads(raw_path.read_text())
    source51 = json.loads((ART / 'model-config51-v1/source-accuracy.json').read_text())
    source38 = json.loads((ART / 'source-only-v1-fresh-optimized.json').read_text())
    source_cases = {family: {case['name']: case for case in data['cases']}
                    for family, data in [('model51', source51), ('regression38', source38)]}
    import torch
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    rows = []
    for case in raw['cases']:
        if 'saved_full_output_file_sha256' not in case:
            rows.append({'name': case['name'], 'family': case['family'], 'output_not_saved': True})
            continue
        family = case['family']
        path = ROOT / 'normal-v5-full-outputs' / f'{family}-{case["name"]}.pt'
        assert digest(path) == case['saved_full_output_file_sha256']
        original = source_cases[family][case['name']]
        assert case['input_sha256'] == original['input_sha256']
        if family == 'model51':
            source_path = ART / 'model-config51-v1/source-full-outputs' / f'{case["name"]}.pt'
            assert digest(source_path) == original['saved_full_output_file_sha256']
        else:
            source_path = ART / 'matched-native-icpx-v1/source-only-paired-v1' / f'{case["name"]}.pt'
        actual = torch.load(path, map_location='cpu', weights_only=True).contiguous()
        expected = torch.load(source_path, map_location='cpu', weights_only=True).contiguous()
        assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
        source_tensor_sha = hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest()
        assert source_tensor_sha == original['output_sha256']
        actual_tensor_sha = hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()
        assert actual_tensor_sha == case['output_sha256']
        mismatches = (actual.view(torch.int16) != expected.view(torch.int16)).count_nonzero().item()
        assert (mismatches == 0) == case['full_output_bitwise']
        max_abs = (actual.float() - expected.float()).abs().max().item()
        assert max_abs == case['source_port_max_abs']
        rows.append({'name': case['name'], 'family': family, 'whole_output_elements': actual.numel(),
            'BF16_bitwise_mismatches': mismatches, 'paired_max_abs': max_abs,
            'input_SHA_matches_genuine_SOURCE': True, 'reference_PASS': case.get('reference_PASS', False),
            'actual_private_route_PASS': case.get('actual_metadata_private_route_PASS', False)})
    assert not torch.xpu.is_initialized()
    report = {'CPU_only': True, 'GPU_initialized': False, 'runtime_actual_exit': actual_exit,
        'raw_sha256': digest(raw_path), 'cases': rows,
        'whole_output_elements': sum(row.get('whole_output_elements', 0) for row in rows),
        'BF16_bitwise_mismatches': sum(row.get('BF16_bitwise_mismatches', 0) for row in rows),
        'whole_saved_output_pairs_checked': sum('whole_output_elements' in row for row in rows),
        'not_full_acceptance_if_runtime_failed_or_cases_missing': True}
    if actual_exit == 0:
        assert len(rows) == 89 and report['whole_output_elements'] == 1884094464
        assert report['BF16_bitwise_mismatches'] == 0
    (ROOT / 'normal-v5-89-full-output-bitwise-CPU-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value for name, value in report.items() if name != 'cases'}), flush=True)


if __name__ == '__main__':
    main()

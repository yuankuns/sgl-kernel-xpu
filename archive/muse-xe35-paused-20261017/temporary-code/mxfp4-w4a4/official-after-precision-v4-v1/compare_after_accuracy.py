import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    options = parser.parse_args()
    import torch
    from input_forensics import tensor_sha256

    torch.set_num_threads(16)
    evidence = Path(options.evidence)
    baseline_reports = json.loads((evidence.parent / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    rounds = [json.loads((evidence / f'round-{round_id}-after.json').read_text()) for round_id in range(4)]
    report = {'scope': 'verified fresh icpx genuine SOURCE versus normal official canonical-v4-icpx AFTER; GPU8 accuracy-only',
              'independent_FP64': 'max_rows4/expert/all columns; not every production row', 'performance': 'not run', 'shapes': {}}
    failed = False
    for index, expected in enumerate(baseline_reports):
        baseline = torch.load(expected['output_file'], map_location='cpu', weights_only=True, mmap=True)
        assert tensor_sha256(baseline) == expected['full_valid_output']
        comparisons = {}
        for round_id, records in enumerate(rounds):
            current = records[index]
            assert current['shape'] == expected['shape'] and current['inputs'] == expected['inputs']
            actual = torch.load(current['output_file'], map_location='cpu', weights_only=True, mmap=True)
            assert actual.shape == baseline.shape and actual.dtype == baseline.dtype
            assert tensor_sha256(actual) == current['full_valid_output']
            different = 0
            violations = 0
            maximum = 0.0
            original_gate_failed = False
            for start in range(0, baseline.shape[0], 256):
                expected_chunk = baseline[start:start + 256]
                actual_chunk = actual[start:start + 256]
                different += torch.count_nonzero(actual_chunk.view(torch.int16) != expected_chunk.view(torch.int16)).item()
                expected_float = expected_chunk.float()
                actual_float = actual_chunk.float()
                delta = (actual_float - expected_float).abs()
                violations += torch.count_nonzero(~torch.isclose(actual_float, expected_float, rtol=0.005, atol=0.005)).item()
                maximum = max(maximum, delta.max().item())
                try:
                    torch.testing.assert_close(actual_chunk, expected_chunk, rtol=0.005, atol=0.005)
                except AssertionError:
                    original_gate_failed = True
            equal_hash = current['full_valid_output'] == expected['full_valid_output']
            comparisons[str(round_id)] = {'original_strict005_pass': not original_gate_failed, 'violating_elements': violations,
                'bitwise_different_elements': different, 'max_abs': maximum, 'checked_elements': baseline.numel(),
                'full_hash_equal': equal_hash, 'independent_reference_pass': current['independent_reference_pass']}
            failed |= original_gate_failed or different != 0 or not equal_hash or not current['independent_reference_pass']
        report['shapes'][expected['shape']] = {'fresh_SOURCE': expected, 'AFTER_rounds': [records[index] for records in rounds],
                                              'full_element_comparisons': comparisons}
    report['strict005_and_full_bitwise_pass'] = not failed
    (evidence / 'source-after-paired-precision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

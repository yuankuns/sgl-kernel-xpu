import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    options = parser.parse_args()
    import torch

    torch.set_num_threads(16)
    evidence = Path(options.evidence)
    all_rounds = [json.loads((evidence / f'round-{round_id}-{kind}.json').read_text()) for round_id, kind in [(0, 'historical'), (1, 'fresh'), (2, 'fresh'), (3, 'historical')]]
    report = {'scope': 'B8/ISL4096/OSL16/TP1/EP1 SOURCE historical versus fresh icpx only; not PORT/AFTER',
              'independent_FP64': 'original max_rows4/expert/all columns, not every production row', 'performance': 'not run', 'shapes': {}}
    failed = False
    for index, baseline_report in enumerate(all_rounds[0]):
        shape = baseline_report['shape']
        baseline = torch.load(baseline_report['output_file'], map_location='cpu', weights_only=True, mmap=True)
        comparisons = {}
        for round_id in [1, 2, 3]:
            current = all_rounds[round_id][index]
            assert current['shape'] == shape and current['inputs'] == baseline_report['inputs']
            actual = torch.load(current['output_file'], map_location='cpu', weights_only=True, mmap=True)
            assert actual.shape == baseline.shape and actual.dtype == baseline.dtype
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
            comparisons[str(round_id)] = {'original_strict005_pass': not original_gate_failed, 'violating_elements': violations,
                'bitwise_different_elements': different, 'max_abs': maximum, 'checked_elements': baseline.numel(),
                'full_hash_equal': current['full_valid_output'] == baseline_report['full_valid_output']}
            failed |= original_gate_failed
        report['shapes'][shape] = {'rounds': [rounds[index] for rounds in all_rounds], 'full_element_comparisons': comparisons}
    report['strict005_pass'] = not failed
    (evidence / 'source-paired-precision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'strict005_pass': not failed, 'shapes': {shape: entry['full_element_comparisons'] for shape, entry in report['shapes'].items()}}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

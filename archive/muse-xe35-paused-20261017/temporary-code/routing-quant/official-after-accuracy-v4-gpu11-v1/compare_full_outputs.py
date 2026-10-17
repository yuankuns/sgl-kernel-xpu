import json
import os
from pathlib import Path

import torch


def main():
    directory = Path(__file__).resolve().parent
    source = directory.parent / 'historical-fresh-source-precision-v1'
    assert sorted(os.sched_getaffinity(0)) == list(range(72, 80))
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    before = json.loads((source / 'round-1-fresh.json').read_text())
    after = json.loads((directory / 'round-0-after.json').read_text())
    assert len(before) == len(after) == 6
    report = []
    failed = False
    for baseline, current in zip(before, after, strict=True):
        assert baseline['shape'] == current['shape']
        original = torch.load(baseline['accuracy']['output'], map_location='cpu', weights_only=True, mmap=True)
        actual = torch.load(current['accuracy']['output'], map_location='cpu', weights_only=True, mmap=True)
        assert actual.keys() == original.keys()
        fields = ['inputs'] if 'weights' in original else ['input_sha256', 'row_map_sha256', 'scale_map_sha256']
        for field in fields:
            assert baseline['accuracy'][field] == current['accuracy'][field]
        comparisons = {}
        for name in actual:
            assert actual[name].shape == original[name].shape and actual[name].dtype == original[name].dtype
            different = int(torch.count_nonzero(actual[name] != original[name]))
            comparisons[name] = {'bitwise_equal': bool(torch.equal(actual[name].view(torch.uint8), original[name].view(torch.uint8))),
                                 'different_elements': different, 'checked_elements': original[name].numel()}
            failed |= not comparisons[name]['bitwise_equal']
        original_gates = {}
        for name in actual:
            try:
                torch.testing.assert_close(actual[name], original[name], rtol=0, atol=2.5e-4 if name == 'weights' else 0)
                passed = True
            except AssertionError:
                passed = False
            original_gates[name] = passed
        report.append({'shape': current['shape'], 'original_precision_gates': original_gates,
                       'all_valid_full_outputs_SOURCE_bit_comparisons': comparisons,
                       'output_scope': current['accuracy'].get('output_scope', 'every route id/weight element')})
    (directory / 'full-SOURCE-AFTER-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

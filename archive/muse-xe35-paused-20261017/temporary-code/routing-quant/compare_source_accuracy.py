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
    rounds = [json.loads((evidence / f'round-{round_id}-{kind}.json').read_text())
              for round_id, kind in [(0, 'historical'), (1, 'fresh'), (2, 'fresh'), (3, 'historical')]]
    assert all(len(entries) == 6 for entries in rounds)
    result = {'scope': 'B8/ISL4096/OSL16/TP1/EP1 historical genuine SOURCE versus fresh icpx SOURCE only',
              'performance': 'not run', 'PORT_AFTERMATCH': 'not run', 'shapes': {}}
    for index, baseline in enumerate(rounds[0]):
        original = torch.load(baseline['accuracy']['output'], map_location='cpu', weights_only=True, mmap=True)
        comparisons = {}
        for round_id in [1, 2, 3]:
            entry = rounds[round_id][index]
            assert entry['shape'] == baseline['shape']
            actual = torch.load(entry['accuracy']['output'], map_location='cpu', weights_only=True, mmap=True)
            assert actual.keys() == original.keys()
            if 'weights' in original:
                assert entry['accuracy']['inputs'] == baseline['accuracy']['inputs']
                torch.testing.assert_close(actual['ids'], original['ids'], atol=0, rtol=0)
                torch.testing.assert_close(actual['weights'], original['weights'], atol=2.5e-4, rtol=0)
                comparisons[str(round_id)] = {'ids_exact': True, 'weights_atol': 2.5e-4, 'weights_rtol': 0,
                                              'all_elements': original['weights'].numel()}
            else:
                for field in ['input_sha256', 'row_map_sha256', 'scale_map_sha256']:
                    assert entry['accuracy'][field] == baseline['accuracy'][field]
                for name, expected in original.items():
                    value = actual[name]
                    assert value.shape == expected.shape and value.dtype == expected.dtype
                    if value.ndim == 2:
                        for start in range(0, value.shape[0], 256):
                            assert torch.equal(value[start:start + 256], expected[start:start + 256]), name
                    else:
                        assert torch.equal(value, expected), name
                for field in ['q_sha256', 'scale_sha256']:
                    assert entry['accuracy'][field] == baseline['accuracy'][field]
                comparisons[str(round_id)] = {'all_valid_FP4_codes_bitwise': True, 'all_used_group32_scales_including_padding_bitwise': True,
                    'full_maps_counts_tail_layout_bitwise': True, 'checked_codes': original['q'].numel(), 'checked_scales': original['scales'].numel()}
        result['shapes'][baseline['shape']] = {'rounds': [entries[index] for entries in rounds], 'full_element_comparisons': comparisons}
    result['original_precision_pass'] = True
    (evidence / 'source-paired-precision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'all_six_shapes_four_rounds': True, 'precision_only': True}))


if __name__ == '__main__':
    main()

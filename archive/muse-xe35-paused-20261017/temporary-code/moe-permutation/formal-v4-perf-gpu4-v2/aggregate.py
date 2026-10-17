import hashlib
import json
import statistics
from pathlib import Path


def main():
    import torch

    folder = Path(__file__).resolve().parent
    root = folder.parent
    rounds = [json.loads((folder / ('canonical-v4-icpx-formal-gpu4-v1-round' + str(cycle) + '-BAAB-summary.json')).read_text()) for cycle in [1, 2, 3]]
    runs = [[json.loads((folder / ('canonical-v4-icpx-formal-gpu4-v1-round' + str(cycle) + '-BAAB-' + str(index) + '-' + label + '.json')).read_text()) for index, label in enumerate(['source', 'port', 'port', 'source'])] for cycle in [1, 2, 3]]
    SOURCE = json.loads((root / 'source-native-icpx-v1-accuracy/fresh.json').read_text())
    rows = []
    precision_checks = []
    for case_index, source_case in enumerate(SOURCE['cases']):
        source_path = root / 'source-native-icpx-v1-accuracy' / source_case['outputs']
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_case['outputs_sha256']
        baseline = torch.load(source_path, weights_only=True)
        for cycle_index, cycle in enumerate(runs):
            for index, run in enumerate(cycle):
                case = run['cases'][case_index]
                assert case['input_sha256'] == source_case['input_sha256']
                output_file = folder / case['outputs']
                assert hashlib.sha256(output_file.read_bytes()).hexdigest() == case['outputs_sha256']
                actual = torch.load(output_file, weights_only=True)
                assert len(actual) == len(baseline) == 4
                for output_index, (expected, output) in enumerate(zip(baseline, actual)):
                    identical = torch.equal(expected, output)
                    precision_checks.append({'round': cycle_index + 1, 'run': index, 'case': case['case'], 'output': output_index, 'elements': output.numel(), 'equal': identical})
                    assert identical, 'Whole legal output/tail mismatch against validated fresh SOURCE'
        source_samples = [value for cycle in runs for index in [0, 3] for value in cycle[index]['cases'][case_index]['samples_ms']['permutation']]
        port_samples = [value for cycle in runs for index in [1, 2] for value in cycle[index]['cases'][case_index]['samples_ms']['permutation']]
        assert len(source_samples) == len(port_samples) == 180
        source_ms = statistics.median(source_samples)
        port_ms = statistics.median(port_samples)
        change = 100 * (port_ms / source_ms - 1)
        run_rows = [cycle['rows'][case_index] for cycle in rounds]
        stable_positive = all(row['small_stable_positive_regression'] for row in run_rows)
        row = {'case': source_case['case'], 'source_ms': source_ms, 'port_ms': port_ms, 'change_pct': change,
               'source_MAD_ms': statistics.median([abs(value - source_ms) for value in source_samples]),
               'port_MAD_ms': statistics.median([abs(value - port_ms) for value in port_samples]),
               'round_change_pct': [row['change_pct'] for row in run_rows],
               'round_SOURCE_drift_pct': [row['source_drift_pct'] for row in run_rows],
               'round_AFTER_drift_pct': [row['port_drift_pct'] for row in run_rows],
               'small_stable_positive_all_rounds': stable_positive,
               'any_round_positive_stability': any(row['small_stable_positive_regression'] for row in run_rows),
               'needs_review': abs(change) > 5 or stable_positive or any(row['needs_retest'] or row['small_stable_positive_regression'] or abs(row['port_drift_pct']) > 5 for row in run_rows),
               'raw_samples_SOURCE_ms': source_samples, 'raw_samples_AFTER_ms': port_samples, 'outliers_removed': 0}
        rows.append(row)
    result = {'round_count': 3, 'GPU': 4, 'rows': rows, 'accuracy_exact_full_tail_passed': True,
              'whole_tensor_checks_against_validated_fresh_SOURCE': precision_checks,
              'cross_run_whole_tensor_checks': sum(len(cycle['full_tensor_bitwise_checks']) for cycle in rounds),
              'needs_review': any(row['needs_review'] for row in rows), 'performance_accepted': False,
              'all_eight_cards_parallel_verified': False, 'coordinator_review_pending': True,
              'prefill_contract': '262144 pairs rejected in all twelve normal native runs; not a prefill fastpath performance claim'}
    (folder / 'aggregate-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if not key.startswith('raw_samples')}), flush=True)
    raise SystemExit(3 if result['needs_review'] else 0)


if __name__ == '__main__':
    main()

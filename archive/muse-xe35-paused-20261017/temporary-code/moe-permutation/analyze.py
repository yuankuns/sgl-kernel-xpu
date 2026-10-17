import argparse
import csv
import json
import statistics
from pathlib import Path


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument('round', nargs='?', default='BAAB')
    args = parser.parse_args()
    import torch

    root = Path(__file__).resolve().parent
    diagnostic = args.round.startswith('kernel-diagnostic')
    labels = ['source', 'diagnostic-port', 'diagnostic-port', 'source'] if diagnostic else ['source', 'port', 'port', 'source']
    runs = [json.loads((root / f'{args.round}-{index}-{label}.json').read_text()) for index, label in enumerate(labels)]
    assert all((root / f'{args.round}-{index}-{label}.exit').read_text().strip() == '0' for index, label in enumerate(labels))
    bitwise = root.name == 'moe-permutation'
    accuracy_only = all(run['scope'].get('accuracy_only', False) for run in runs)
    expected_gpu = '10' if diagnostic and bitwise or accuracy_only else '4'
    assert all(run['gpu_mask'] == expected_gpu for run in runs), 'source/port must share the current leased GPU'
    assert len({run['device_properties'] for run in runs}) == 1
    assert len({run['torch'] for run in runs}) == 1
    assert all(run['scope']['serving_batch'] == 8 and run['scope']['isl'] == 4096
               and run['scope']['osl'] == 16 and run['scope']['tp'] == 1 and run['scope']['ep'] == 1
               for run in runs)
    assert len({len(run['cases']) for run in runs}) == 1
    rows = []
    for case_index, first in enumerate(runs[0]['cases']):
        cases = [run['cases'][case_index] for run in runs]
        assert len({case['input_sha256'] for case in cases}) == 1
        if not bitwise:
            assert len({case['weight_sha256'] for case in cases}) == 1
        outputs = [torch.load(root / case['outputs'], weights_only=True) for case in cases]
        for other in outputs[1:]:
            if bitwise:
                assert all(torch.equal(first_output, other_output) for first_output, other_output in zip(outputs[0], other))
            else:
                for name, first_output in outputs[0].items():
                    torch.testing.assert_close(first_output, other[name], atol=5e-3, rtol=5e-3)
        if accuracy_only:
            continue
        for name in first['samples_ms']:
            source_samples = cases[0]['samples_ms'][name] + cases[3]['samples_ms'][name]
            port_samples = cases[1]['samples_ms'][name] + cases[2]['samples_ms'][name]
            source_ms = statistics.median(source_samples)
            port_ms = statistics.median(port_samples)
            change = 100 * (port_ms / source_ms - 1)
            drift = 100 * (cases[3]['medians_ms'][name] / cases[0]['medians_ms'][name] - 1)
            row = {'case': first.get('case', f'M8-N{first.get("n")}-K8192'), 'path': name,
                   'source_ms': source_ms, 'port_ms': port_ms, 'change_pct': change, 'source_drift_pct': drift,
                   'accuracy': 'bitwise_pass' if bitwise else 'atol_rtol_5e-3_pass',
                   'needs_retest': abs(change) > 5 or abs(drift) > 5,
                   'run_medians_ms': [case['medians_ms'][name] for case in cases]}
            rows.append(row)
    result = {'round': args.round, 'order': labels, 'gpu': int(expected_gpu), 'rows': rows,
              'verification_kind': 'port-kernel diagnostic, not official integration' if diagnostic else 'official integration',
              'accuracy_passed': True, 'needs_retest': any(row['needs_retest'] for row in rows),
              'prefill': [run.get('prefill') for run in runs] if bitwise else None,
              'performance_acceptance_eligible': not diagnostic and not accuracy_only and expected_gpu in ['4', '5', '6', '7'],
              'performance_accepted': not diagnostic and not accuracy_only and expected_gpu == '4' and not any(row['needs_retest'] for row in rows),
              'accuracy_only': accuracy_only,
              'historical_non_acceptance_gpu10_performance': diagnostic and expected_gpu == '10'}
    (root / (args.round + '-summary.json')).write_text(json.dumps(result, indent=2) + '\n')
    with (root / (args.round + '-summary.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['case', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f'{row["case"]} {row["path"]}: source={row["source_ms"]:.6f}ms port={row["port_ms"]:.6f}ms '
              f'change={row["change_pct"]:+.2f}% source_drift={row["source_drift_pct"]:+.2f}% {row["accuracy"]}', flush=True)
    print(json.dumps({'needs_retest': result['needs_retest'], 'accuracy_passed': True}), flush=True)
    raise SystemExit(3 if result['needs_retest'] else 0)


if __name__ == '__main__':
    main_xe35()

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ('B0', 'A0', 'A1', 'B1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--prefix', default='diagnostic-perf')
    args = parser.parse_args()
    directory = ROOT / args.component
    rounds = {}
    evidence = {}
    for run in RUNS:
        filename = directory / f'{args.prefix}-{run}.json'
        exit_file = filename.with_suffix('.exit')
        if exit_file.read_text().strip() != '0':
            raise RuntimeError(f'Nonzero benchmark status: {exit_file}')
        data = json.loads(filename.read_text())
        expected_label = 'before' if run.startswith('B') else 'after'
        if (data['label'], data['gpu_affinity'], data['serving_batch'], data['isl'],
                data['osl'], data['tp'], data['ep']) != (expected_label, '11', 8, 4096, 16, 1, 1):
            raise RuntimeError(f'Unmatched workload or device: {filename}')
        if any(result['accuracy'] != 'PASS' for result in data['results']):
            raise RuntimeError(f'Accuracy gate failed: {filename}')
        for result in data['results']:
            if len(result['raw_us']) != data['samples']:
                raise RuntimeError(f'Missing timing samples: {filename}')
            if abs(statistics.median(result['raw_us']) - result['p50_us']) > 1e-9:
                raise RuntimeError(f'Median disagrees with raw samples: {filename}')
        rounds[run] = data
        evidence[filename.name] = hashlib.sha256(filename.read_bytes()).hexdigest()
    for earlier, later in (('B0', 'B1'), ('A0', 'A1')):
        for field in ('libraries', 'package', 'samples', 'calls', 'api', 'timing'):
            if rounds[earlier][field] != rounds[later][field]:
                raise RuntimeError(f'Within-label provenance mismatch: {field}')
    cases = [result['case'] for result in rounds['B0']['results']]
    for run in RUNS:
        if [result['case'] for result in rounds[run]['results']] != cases:
            raise RuntimeError(f'Case mismatch: {run}')
    rows = []
    for index, case in enumerate(cases):
        medians = {run: rounds[run]['results'][index]['p50_us'] for run in RUNS}
        source = statistics.mean(medians[run] for run in ('B0', 'B1'))
        port = statistics.mean(medians[run] for run in ('A0', 'A1'))
        delta = (port / source - 1) * 100
        drift = (medians['B1'] / medians['B0'] - 1) * 100
        row = {
            'case': case, 'source_us': source, 'port_us': port,
            'port_change_percent': delta, 'source_drift_percent': drift,
            'requires_repeat': abs(delta) > 5 or abs(drift) > 5,
        }
        for run in RUNS:
            row[f'{run}_median_us'] = medians[run]
            row[f'{run}_mad_us'] = rounds[run]['results'][index]['mad_us']
        rows.append(row)
    report = {
        'component': args.component, 'gpu': 11, 'round_order': list(RUNS),
        'comparison': 'original optimized source versus upstream port, not pre-optimization baseline',
        'stage': 'kernel diagnostic with real original-package registration and native ABI forwards',
        'official_full_package_acceptance': False,
        'samples_per_run': rounds['B0']['samples'], 'calls_per_sample': rounds['B0']['calls'],
        'method': 'mean of two per-run medians; positive change means slower port; repeat absolute change or source drift over 5%',
        'evidence_sha256': evidence, 'results': rows,
    }
    output = directory / f'{args.prefix}-summary'
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
import statistics


ART = Path(__file__).resolve().parent


def round_summary_xe35(round_index, prefix='perf'):
    labels = [f'{prefix}-r{round_index}-{slot}' for slot in ('B0', 'A1', 'A2', 'B3')]
    runs = [json.loads((ART / f'{label}.json').read_text()) for label in labels]
    assert all((ART / f'{label}.exit').read_text().strip() == '0' for label in labels)
    devices = {run['provenance']['gpu_affinity'] for run in runs}
    assert len(devices) == 1, devices
    rows = [{row['name']: row for row in run['cases']} for run in runs]
    assert all(set(rows[0]) == set(row) for row in rows)
    assert len(rows[0]) == 38 if round_index == 1 else len(rows[0]) >= 8
    results = []
    for name in rows[0]:
        source = statistics.mean([rows[0][name]['median_ms'], rows[3][name]['median_ms']])
        port = statistics.mean([rows[1][name]['median_ms'], rows[2][name]['median_ms']])
        regression = (port / source - 1) * 100
        results.append({'name': name, 'source_ms': source, 'port_ms': port, 'regression_percent': regression,
                        'over_5_percent': regression > 5,
                        'source_endpoint_drift_percent': (rows[3][name]['median_ms'] / rows[0][name]['median_ms'] - 1) * 100,
                        'paired_samples': {label: rows[index][name]['samples_ms'] for index, label in enumerate(labels)}})
    return {'order': labels, 'B': 'source', 'A': 'port', 'gpu_affinity': devices.pop(), 'cases': results,
            'over_5_percent': [row['name'] for row in results if row['over_5_percent']]}


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='perf')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--official', action='store_true')
    args = parser.parse_args()
    if args.diagnostic and args.official:
        parser.error('official and diagnostic are mutually exclusive')
    if not args.prefix or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.prefix):
        parser.error('prefix must be a safe run label')
    summary = {'acceptance_context': 'diagnostic_not_official_full_package_pass' if args.diagnostic else 'official_validation_pending',
               'prefix': args.prefix}
    for round_index in (1, 2):
        labels = [f'{args.prefix}-r{round_index}-{slot}' for slot in ('B0', 'A1', 'A2', 'B3')]
        if all((ART / f'{label}.json').exists() and (ART / f'{label}.exit').exists() for label in labels):
            summary[f'round_{round_index}'] = round_summary_xe35(round_index, args.prefix)
    for side in ('source', 'port'):
        label = f'{args.prefix}-{side}-accuracy' if args.diagnostic or args.official else f'{side}-accuracy'
        path = ART / f'{label}.json'
        if path.exists():
            data = json.loads(path.read_text())
            cases = data['cases']
            summary[f'{side}_accuracy'] = {'cases': len(cases),
                'full_decode_cases': sum(row['query_length'] == 1 and row.get('coverage') == 'full' for row in cases),
                'full_prefill_cases': sum(row['query_length'] == 4096 and row.get('coverage') == 'full' for row in cases),
                'max_abs': max(row.get('max_abs', 0) for row in cases),
                'paired_max_abs': max(row.get('paired_max_abs', 0) for row in cases),
                'complete': (ART / f'{label}.exit').exists() and (ART / f'{label}.exit').read_text().strip() == '0' and len(cases) == 38,
                'gpu_affinity': data.get('provenance', {}).get('gpu_affinity')}
    summary['exits'] = {path.stem: path.read_text().strip() for path in sorted(ART.glob('*.exit'))}
    name = f'{args.prefix}-summary.json' if args.diagnostic or args.official else 'validation-summary.json'
    (ART / name).write_text(json.dumps(summary, indent=2) + '\n')
    for key, value in summary.items():
        print(key, {item: data for item, data in value.items() if item != 'cases'} if key.startswith('round') else value)


if __name__ == '__main__':
    main_xe35()

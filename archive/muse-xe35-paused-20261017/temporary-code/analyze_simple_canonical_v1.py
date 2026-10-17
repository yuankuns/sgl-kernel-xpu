import argparse
import json
import math
import statistics
from pathlib import Path

from canonical_tools_loader_v1 import digest_file


ROOT = Path(__file__).resolve().parent
RUNS = ('B0', 'A0', 'A1', 'B1')


def relative_change(before, after):
    return (after / before - 1) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--validation-version', default='canonical-validation-v4-icpx')
    args = parser.parse_args()
    evidence = ROOT / args.component / args.validation_version
    records = {}
    hashes = {}
    flags = []
    for phase in ('accuracy', 'performance'):
        complete = json.loads((evidence / f'{phase}-execution-complete.json').read_text())
        if complete['actual_exit'] != 0 or complete['gpu'] != ('11' if phase == 'accuracy' else '7'):
            raise RuntimeError('Wrong allocation or unsuccessful execution')
        for label in ('before', 'after'):
            prefix = evidence / f'{phase}-package-{label}-preflight'
            if prefix.with_suffix('.exit').read_text().strip() != '0':
                raise RuntimeError('Normal current-package preflight failed')
            preflight = json.loads(prefix.with_suffix('.log').read_text())
            if not all(preflight[field] for field in ('default_all_verified', 'normal_package_import_verified', 'all_runtime_mapped_dso_verified')):
                raise RuntimeError('Required current-package gate did not pass')
            hashes[prefix.with_suffix('.log').name] = digest_file(prefix.with_suffix('.log'))
    for round_id in range(3):
        records[round_id] = {}
        for run in RUNS:
            prefix = evidence / f'performance-r{round_id}-{run}'
            if prefix.with_suffix('.exit').read_text().strip() != '0':
                raise RuntimeError('Nonzero true performance subprocess status')
            data = json.loads(prefix.with_suffix('.json').read_text())
            label = 'before' if run.startswith('B') else 'after'
            if (data['label'], data['phase'], data['gpu_affinity'], data['serving_batch'], data['isl'], data['osl'], data['tp'], data['ep'], data['samples'], data['calls']) != (label, 'performance', '7', 8, 4096, 16, 1, 1, 101, 128):
                raise RuntimeError('Unmatched stable-card scope/method')
            for result in data['results']:
                samples = result['raw_us']
                if result['accuracy'] != 'PASS' or len(samples) != 101 or any(not math.isfinite(value) or value <= 0 for value in samples):
                    raise RuntimeError('Accuracy failure or incomplete/invalid samples')
                if abs(statistics.median(samples) - result['p50_us']) > 1e-9:
                    raise RuntimeError('Recorded median differs from raw samples')
            records[round_id][run] = data
            hashes[prefix.with_suffix('.json').name] = digest_file(prefix.with_suffix('.json'))
    first = records[0]['B0']
    expected_cases = [result['case'] for result in first['results']]
    for round_id, rounds in records.items():
        for run, data in rounds.items():
            if [result['case'] for result in data['results']] != expected_cases:
                raise RuntimeError('Case coverage/order changed between rounds')
            comparison = records[0]['B0' if run.startswith('B') else 'A0']
            for field in ('package', 'libraries', 'timing', 'api', 'driver_sha256', 'reference_driver_sha256'):
                if data[field] != comparison[field]:
                    raise RuntimeError(f'Provider or method changed within BAAB: {field}')
    rows = []
    for index, case in enumerate(expected_cases):
        per_round = []
        for round_id, rounds in records.items():
            medians = {run: rounds[run]['results'][index]['p50_us'] for run in RUNS}
            source = (medians['B0'] + medians['B1']) / 2
            port = (medians['A0'] + medians['A1']) / 2
            result = {'round': round_id, 'source_us': source, 'port_us': port, 'port_change_pct': relative_change(source, port), 'source_drift_pct': relative_change(medians['B0'], medians['B1']), 'port_drift_pct': relative_change(medians['A0'], medians['A1']), 'run_medians_us': medians, 'run_mad_us': {run: rounds[run]['results'][index]['mad_us'] for run in RUNS}}
            per_round.append(result)
            if result['port_change_pct'] > 5 or abs(result['source_drift_pct']) > 5 or abs(result['port_drift_pct']) > 5:
                flags.append({'case': case, **result})
        source = statistics.mean(item['source_us'] for item in per_round)
        port = statistics.mean(item['port_us'] for item in per_round)
        rows.append({'case': case, 'source_us': source, 'port_us': port, 'port_change_pct': relative_change(source, port), 'all_rounds': per_round})
    report = {'component': args.component, 'performance_gpu': 7, 'accuracy_gpu': 11, 'round_order': list(RUNS), 'all_rounds_preserved': True, 'comparison': 'original optimized source versus upstream port, not pre-optimization baseline', 'stage': 'current normal official package with default-all build and fresh current-graph install; source comparator uses original native kernels only', 'evidence_sha256': hashes, 'results': rows, 'large_change_or_drift_review_flags': flags, 'performance_acceptance_complete': False, 'acceptance_note': '5% flags trigger investigation; absence of these flags is not proof of zero regression. Coordinator reviews all per-round medians/MAD and reproducibility, including smaller stable regressions.'}
    path = evidence / 'performance-all-rounds-summary.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite analysis evidence')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': str(path), 'case_count': len(rows), 'review_flags': len(flags), 'performance_acceptance_complete': False}))


if __name__ == '__main__':
    main()

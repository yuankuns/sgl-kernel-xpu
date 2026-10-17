import argparse
import json
from pathlib import Path

from check_preopt import BASE
from summarize import round_summary_xe35


ART = Path(__file__).resolve().parent


def accuracy_gate(prefix):
    cases_by_side = {}
    for side in ('preopt', 'port'):
        label = f'{prefix}-{side}-accuracy'
        assert (ART / f'{label}.exit').read_text().strip() == '0', label
        data = json.loads((ART / f'{label}.json').read_text())
        assert data['provider'] == side and data['preopt_base_sha'] == BASE
        assert data['provenance']['gpu_affinity'] == '5'
        cases = {record['name']: record for record in data['cases']}
        assert len(cases) == 38
        assert sum(record['query_length'] == 4096 for record in cases.values()) == 6
        assert sum(record['query_length'] == 1 for record in cases.values()) == 32
        for record in cases.values():
            assert record['accuracy'] == 'PASS' and record['coverage'] == 'full'
            assert record['reference_device'] == 'cpu'
            assert record['atol'] == record['rtol'] == 5e-3
            assert record['checked_elements'] == record['requests'] * record['query_length'] * 128 * 64
        cases_by_side[side] = data
    assert cases_by_side['preopt']['canonical_source_digest'] == cases_by_side['port']['canonical_source_digest']
    assert {record['name'] for record in cases_by_side['preopt']['cases']} == {record['name'] for record in cases_by_side['port']['cases']}
    return cases_by_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--accuracy-only', action='store_true')
    args = parser.parse_args()
    assert args.prefix and all(character.isascii() and (character.isalnum() or character in '_-') for character in args.prefix)
    accuracy = accuracy_gate(args.prefix)
    report = {'comparison': 'real historical preopt ab3a26c to canonical official port; not optimized-source to port',
              'preopt_base_sha': BASE, 'accuracy_complete': True,
              'canonical_source_digest': accuracy['port']['canonical_source_digest'],
              'cpu_reference_wall_seconds': {side: sum(record['cpu_reference_wall_seconds'] for record in data['cases'])
                                             for side, data in accuracy.items()}}
    if not args.accuracy_only:
        for round_index in (1, 2):
            labels = [f'{args.prefix}-r{round_index}-{slot}' for slot in ('B0', 'A1', 'A2', 'B3')]
            if not all((ART / f'{label}.exit').exists() for label in labels):
                assert round_index == 2
                continue
            for index, label in enumerate(labels):
                data = json.loads((ART / f'{label}.json').read_text())
                assert data['provider'] == ('preopt' if index in (0, 3) else 'port')
                assert data['canonical_source_digest'] == report['canonical_source_digest']
            summary = round_summary_xe35(round_index, args.prefix)
            summary['B'] = 'preopt ab3a26c'
            for record in summary['cases']:
                record['preopt_ms'] = record.pop('source_ms')
                record['speedup'] = record['preopt_ms'] / record['port_ms']
            report[f'round_{round_index}'] = summary
    (ART / f'{args.prefix}-preopt-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

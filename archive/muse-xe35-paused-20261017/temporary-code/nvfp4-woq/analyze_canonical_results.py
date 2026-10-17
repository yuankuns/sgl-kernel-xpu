import argparse
import json
import pathlib
import statistics
import re

import canonical_check

parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, choices=('accuracy', 'performance'))
parser.add_argument('--run', required=True)
parser.add_argument('--version', default=canonical_check.DEFAULT_VERSION)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
canonical_check.select_version(args.version)
directory = root / args.version / 'results' / args.run
groups = {}
paths = sorted(directory.glob('round*-phase*.json'))
expected_rounds = 1 if args.phase == 'accuracy' else 3
assert len(paths) == expected_rounds * 4
for path in paths:
    payload = json.loads(path.read_text())
    assert payload['canonical_version'] == args.version
    assert payload['accuracy_only'] == (args.phase == 'accuracy')
    assert payload['gpu'] == ('9' if args.phase == 'accuracy' else '5')
    assert len(payload['results']) == 4
    for result in payload['results']:
        assert result['output']['all_elements_checked']
        assert all(record['all_elements_checked'] for record in result['scales'].values())
        if args.phase == 'accuracy':
            assert result['samples_us'] == []
        else:
            assert len(result['samples_us']) == 20 and all(value > 0 for value in result['samples_us'])
        key = (tuple(result['shape']), result['mode'])
        groups.setdefault(key, {}).setdefault(result['round'], {})[result['phase']] = result
assert len(groups) == 4
summaries = []
regressions = []
for (shape, mode), rounds in groups.items():
    assert sorted(rounds) == list(range(expected_rounds))
    ratios = []
    for phases in rounds.values():
        assert sorted(phases) == [0, 1, 2, 3]
        assert [phases[index]['label'] for index in range(4)] == ['port', 'source', 'source', 'port']
        assert all(phases[index]['inputs'] == phases[0]['inputs'] for index in range(4))
        if args.phase == 'performance':
            port = phases[0]['samples_us'] + phases[3]['samples_us']
            source = phases[1]['samples_us'] + phases[2]['samples_us']
            ratios.append(statistics.median(port) / statistics.median(source))
    regression = statistics.median(ratios) > 1.05 if ratios else None
    summary = {'shape': shape, 'mode': mode, 'gpu': 9 if args.phase == 'accuracy' else 5, 'phase': args.phase, 'ratios_port_source': ratios, 'regression_gt_5pct': regression, 'accuracy_verified': True, 'performance_verified': args.phase == 'performance' and not regression, 'all_output_exact': all(result['output']['exact'] for phases in rounds.values() for result in phases.values()), 'max_abs': max(result['output']['max_abs'] for phases in rounds.values() for result in phases.values())}
    summaries.append(summary)
    if regression:
        regressions.append(summary)
(directory / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
print(json.dumps(summaries, indent=2))
if regressions:
    raise RuntimeError('GPU5 >5% performance regression: repeat/diagnose required, not acceptance')

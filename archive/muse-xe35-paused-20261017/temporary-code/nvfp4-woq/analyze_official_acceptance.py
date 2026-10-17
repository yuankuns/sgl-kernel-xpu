import json
import pathlib
import statistics
import sys


artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq')
directory = artifact / 'official-results' / sys.argv[1]
groups = {}
for path in sorted(directory.glob('round*-phase*.json')):
    for result in json.loads(path.read_text())['results']:
        key = (tuple(result['shape']), result['mode'])
        groups.setdefault(key, {}).setdefault(result['round'], {})[result['phase']] = result
summaries = []
regressions = []
for (shape, mode), rounds in groups.items():
    ratios = []
    for round_index, phases in sorted(rounds.items()):
        assert sorted(phases) == [0, 1, 2, 3]
        assert [phases[index]['label'] for index in range(4)] == ['port', 'source', 'source', 'port']
        port = phases[0]['samples_us'] + phases[3]['samples_us']
        source = phases[1]['samples_us'] + phases[2]['samples_us']
        if source:
            ratios.append(statistics.median(port) / statistics.median(source))
    regression = statistics.median(ratios) > 1.05 if ratios else None
    summary = {'shape': shape, 'mode': mode, 'ratios_port_source': ratios,
               'regression_gt_5pct': regression, 'diagnostic_only': False, 'official_acceptance': True}
    summaries.append(summary)
    if regression:
        regressions.append(summary)
    print(json.dumps(summary), flush=True)
assert len(groups) == 4
expected_rounds = 1 if sys.argv[1].endswith('-tails') else 3
assert all(len(rounds) == expected_rounds for rounds in groups.values())
assert len(list(directory.glob('round*-phase*.json'))) == 4 * expected_rounds
(directory / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
if regressions:
    raise RuntimeError(f'>5% regression: repeat/diagnose required, not a pass: {regressions}')

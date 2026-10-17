import json
from pathlib import Path


root = Path(__file__).resolve().parent
report = json.loads((root / 'long_graph_summary.json').read_text())
groups = {}
for case in report['per_BAAB_round']:
    key = (case['environment'], case['calls'], tuple(case['case']))
    groups.setdefault(key, []).append(case)
stable_positive = []
for key, cases in groups.items():
    assert len(cases) == 2
    if all(case['delta_pct'] > 0 for case in cases):
        stable_positive.append({'environment': key[0], 'calls': key[1], 'case': key[2],
                                'round_delta_pct': [case['delta_pct'] for case in cases],
                                'AA_drift_pct': [case['source_AA_drift_pct'] for case in cases],
                                'BB_drift_pct': [case['port_BB_drift_pct'] for case in cases]})
result = {'performance_approved': False, 'diagnostic_only': True,
          'stable_positive_two_round_observations': stable_positive,
          'review': [
              'baseline128 weight/widen has +1.775% and +1.449% with small AA/BB drift; retain small consistent regression, not automatic PASS',
              'same weight case baseline256 is -0.030% and -0.218%; not evidence of graph-length-independent kernel regression',
              'baseline128 weight host enqueue span source145.861/port144.403us and graph submit6.471/6.486us do not explain the positive event delta by larger median port CPU submission',
              'old40-call +6.21% and +5.34% had no CPU-span telemetry; long graph observations cannot establish their root cause retrospectively',
              'long graph reduces fixed-host-cost fraction as a plausible mechanism, not a measured exclusive-kernel decomposition; no CPU span subtraction',
              'GNU libgomp is loaded; KMP_BLOCKTIME=0 is not proof of a controlled or effective spin change',
              'no GPU frequency telemetry: no DVFS attribution; old samples/flags and canonical normal-new-package acceptance remain pending'
          ]}
(root / 'long_graph_round_review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

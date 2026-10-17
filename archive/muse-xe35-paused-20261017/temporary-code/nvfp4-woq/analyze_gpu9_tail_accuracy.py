import json
import pathlib

root = pathlib.Path(__file__).resolve().parent
directory = root / 'native-diagnostic/results/grf512-accuracy-only-tails'
labels = ('port', 'source', 'source', 'port')
records = []
for phase, label in enumerate(labels):
    report = json.loads((directory / f'round0-phase{phase}-{label}.json').read_text())
    assert len(report['results']) == 4
    for result in report['results']:
        assert result['samples_us'] == [], 'GPU9 accuracy must not collect timing samples'
        assert result['output']['all_elements_checked']
        assert result['label'] == label and result['phase'] == phase
        records.append(result)
summary = {'gpu': 9, 'accuracy_only': True, 'timing_samples': 0, 'records': len(records), 'full_output_max_abs': max(record['output']['max_abs'] for record in records), 'all_output_exact': all(record['output']['exact'] for record in records), 'official_acceptance': False, 'formal_performance_gpu': 5, 'main_gpu5_granted': False}
(root / 'grf512-accuracy-only-tails-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

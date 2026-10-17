import json
from pathlib import Path
import statistics


directory = Path(__file__).resolve().parent
phases = [json.loads((directory / f'phase-{phase}-{kind}.json').read_text())
          for phase, kind in enumerate(['source', 'after', 'after', 'source'] * 3)]
assert all(len(records) == 5 for records in phases)
results = []
hold = False
for index in range(5):
    records = [phase[index] for phase in phases]
    assert len({record['shape'] for record in records}) == 1
    assert all(record['input_hashes'] == records[0]['input_hashes'] and record['full_valid_output_hash'] == records[0]['full_valid_output_hash'] for record in records)
    assert all(len(record['raw_per_call_ms']) == 31 and record['calls_per_graph'] == 8 and record['warmup_graph_replays'] == 5 for record in records)
    source = [sample for record in records if record['kind'] == 'source' for sample in record['raw_per_call_ms']]
    after = [sample for record in records if record['kind'] == 'after' for sample in record['raw_per_call_ms']]
    source_median = statistics.median(source)
    after_median = statistics.median(after)
    source_mad = statistics.median(abs(value - source_median) for value in source)
    after_mad = statistics.median(abs(value - after_median) for value in after)
    changes = []
    drift = []
    for cycle in range(3):
        batch = records[cycle * 4:cycle * 4 + 4]
        before_ms = statistics.median(batch[0]['raw_per_call_ms'] + batch[3]['raw_per_call_ms'])
        after_ms = statistics.median(batch[1]['raw_per_call_ms'] + batch[2]['raw_per_call_ms'])
        changes.append((after_ms / before_ms - 1) * 100)
        drift.append({'cycle': cycle, 'source_BB_drift_pct': (batch[3]['median_ms'] / batch[0]['median_ms'] - 1) * 100,
                      'after_AA_drift_pct': (batch[2]['median_ms'] / batch[1]['median_ms'] - 1) * 100})
    change = (after_median / source_median - 1) * 100
    stable_positive = all(value > 0 for value in changes) and after_median - source_median > 3 * (source_mad + after_mad)
    drift_hold = any(abs(record[field]) > 5 for record in drift for field in ['source_BB_drift_pct', 'after_AA_drift_pct'])
    shape_hold = change > 5 or stable_positive or drift_hold
    hold |= shape_hold
    results.append({'shape': records[0]['shape'], 'source_median_ms': source_median, 'after_median_ms': after_median,
                    'source_MAD_ms': source_mad, 'after_MAD_ms': after_mad, 'change_pct': change,
                    'three_BAAB_cycle_changes_pct': changes, 'within_cycle_drift': drift,
                    'regression_above_5_pct': change > 5, 'stable_positive_hold': stable_positive,
                    'drift_above_5_pct_hold': drift_hold, 'requires_main_review_retest_localization': shape_hold})
report = {'all_five_shapes_three_BAAB_complete': True, 'B': 'genuine fresh SOURCE', 'A': 'genuine normal canonical-v4 AFTER',
          'performance_gate_passed': not hold, 'shapes': results,
          'no_math_or_dispatch_changes': True, 'no_automatic_ungranted_retest': True}
(directory / 'performance-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(5 if hold else 0)

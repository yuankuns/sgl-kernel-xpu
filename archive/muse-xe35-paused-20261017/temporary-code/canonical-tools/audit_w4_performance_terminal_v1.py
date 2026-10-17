import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / 'mxfp4-w4a4/official-performance-v4-gpu6-v2'


def main():
    analysis = json.loads((DIRECTORY / 'performance-analysis.json').read_text())
    phases = []
    hashes = {}
    for phase, kind in enumerate(['source', 'after', 'after', 'source'] * 3):
        path = DIRECTORY / f'phase-{phase}-{kind}.json'
        assert int(path.with_suffix('.exit').read_text()) == 0
        records = json.loads(path.read_text())
        assert len(records) == 5
        for record in records:
            assert record['phase'] == phase and record['kind'] == kind
            assert record['calls_per_graph'] == 8 and record['measured_repeats'] == 31
            assert record['warmup_graph_replays'] == 5
            assert record['genuine_replay_NaN_sentinel_full_valid_output_proof']
            assert len(record['raw_graph_ms']) == len(record['raw_per_call_ms']) == 31
            assert all(math.isfinite(value) and value > 0 for value in record['raw_graph_ms'])
            assert all(math.isclose(graph_ms / 8, per_call_ms, rel_tol=1e-12)
                       for graph_ms, per_call_ms in zip(record['raw_graph_ms'], record['raw_per_call_ms']))
            assert statistics.median(record['raw_per_call_ms']) == record['median_ms']
        phases.append(records)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    results = []
    for shape_index, original in enumerate(analysis['shapes']):
        records = [phase[shape_index] for phase in phases]
        assert all(record['shape'] == original['shape'] for record in records)
        assert all(record['input_hashes'] == records[0]['input_hashes'] and
                   record['full_valid_output_hash'] == records[0]['full_valid_output_hash'] for record in records)
        source = [value for record in records if record['kind'] == 'source' for value in record['raw_per_call_ms']]
        after = [value for record in records if record['kind'] == 'after' for value in record['raw_per_call_ms']]
        change = (statistics.median(after) / statistics.median(source) - 1) * 100
        assert change == original['change_pct']
        results.append({'shape': original['shape'], 'change_pct': change,
                        'raw_samples_per_role': len(source), 'original_gate': original})
    assert not analysis['performance_gate_passed']
    for filename in ('host-launch.exit', 'gpu-runtime.exit', 'performance-analysis.exit'):
        assert int((DIRECTORY / filename).read_text()) == 5
    for pid in (3269093, 3269094):
        assert not Path(f'/proc/{pid}').exists()
    report = {'actual_exit': 0, 'independent_raw_device_timing_recalculation': True,
              'performance_acceptance': False, 'original_exit_5_retained': True,
              'all_12_phase_exits_zero': True, 'all_60_records_input_and_output_hashes_equal': True,
              'output_hashes_recomputed_from_tensor_bytes_in_this_audit': False,
              'independent_FP64_scope': 'at most four rows per expert, not every production row',
              'host_owners_absent': [3269093, 3269094],
              'container_owner_and_GPU_lock_drain_not_proven_by_this_audit': True,
              'raw_JSON_sha256': hashes, 'shapes': results}
    with (ROOT / 'W4_PERFORMANCE_MAIN_TERMINAL_AUDIT_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'performance_acceptance': False,
                      'shapes': [{'shape': entry['shape'], 'change_pct': entry['change_pct']} for entry in results]}))


if __name__ == '__main__':
    main()

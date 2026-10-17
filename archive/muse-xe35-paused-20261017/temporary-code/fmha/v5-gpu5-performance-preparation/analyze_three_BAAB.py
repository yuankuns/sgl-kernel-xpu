import argparse
import hashlib
import json
from pathlib import Path
import statistics
import math


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_path(value):
    return Path(value.replace('/root/', '/home/syk/', 1)) if value.startswith('/root/') else Path(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--comparison', choices=('source-port', 'preopt-source'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    runs = {}
    inputs = {}
    identity = None
    for round_number in (1, 2, 3):
        for slot in ('B0', 'A1', 'A2', 'B3'):
            path = args.input_dir / f'{args.comparison}-r{round_number}-{slot}.json'
            raw = json.loads(path.read_text())
            assert raw['comparison'] == args.comparison and raw['round'] == round_number and raw['slot'] == slot
            expected_role = ('port' if slot.startswith('B') else 'source') if args.comparison == 'source-port' else ('source' if slot.startswith('B') else 'preopt')
            assert raw['role'] == expected_role
            current_identity = {field: raw[field] for field in ('driver_sha256', 'grant_sha256', 'canonical_manifest_sha256',
                'source_digest', 'finalseal_sha256', 'baseline_audit_sha256', 'model51_plan_sha256', 'compiler_policy_sha256')}
            if identity is None:
                identity = current_identity
            assert current_identity == identity
            assert raw['all89_input_SHA_capture_nodes_wholebits_samples_PASS'] and len(raw['cases']) == 89
            assert raw['workload'] == {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}
            runs[round_number, slot] = {(case['family'], case['name']): case for case in raw['cases']}
            assert len(runs[round_number, slot]) == 89
            assert sum(case['family'] == 'model51' for case in raw['cases']) == 51
            assert sum(case['family'] == 'representative38' for case in raw['cases']) == 38
            inputs[str(path)] = digest(path)
            for case in raw['cases']:
                assert len(case['samples_ms']) == 21 and all(math.isfinite(sample) and sample > 0 for sample in case['samples_ms'])
                assert (case['warmup'], case['iterations'], case['graph_replays']) == (5, 21, 8)
                assert case['wholebits_before_timing_PASS'] and case['wholebits_after_timing_PASS']
                assert len(case['capture_actual_registered_fwd_nodes']) == 8
                graph = host_path(case['actual_graph_debug_dump'])
                assert graph.suffix == '.dot' and digest(graph) == case['actual_graph_debug_dump_sha256']
                assert digest(host_path(case['saved_output_file'])) == case['saved_output_file_sha256']
    keys = set(runs[1, 'B0'])
    assert all(set(run) == keys for run in runs.values())
    records = []
    for family, name in sorted(keys):
        rows = []
        baseline_inputs = runs[1, 'B0'][family, name]['input_sha256']
        role_output_hashes = {}
        for round_number in (1, 2, 3):
            slots = {slot: runs[round_number, slot][family, name] for slot in ('B0', 'A1', 'A2', 'B3')}
            assert all(case['input_sha256'] == baseline_inputs for case in slots.values())
            for slot, case in slots.items():
                role = 'after' if slot.startswith('B') else 'before'
                if role not in role_output_hashes:
                    role_output_hashes[role] = case['output_sha256']
                assert role_output_hashes[role] == case['output_sha256'], 'within-role whole-output drift across BAAB/rounds'
            medians = {slot: statistics.median(case['samples_ms']) for slot, case in slots.items()}
            before = statistics.median(slots['A1']['samples_ms'] + slots['A2']['samples_ms'])
            after = statistics.median(slots['B0']['samples_ms'] + slots['B3']['samples_ms'])
            hashes = {case['output_sha256'] for case in slots.values()}
            if args.comparison == 'source-port':
                assert len(hashes) == 1, 'SOURCE/PORT wholebits mismatch invalidates timing acceptance'
            rows.append({'round': round_number, 'before_median_ms': before, 'after_median_ms': after,
                'after_delta_percent': (after / before - 1) * 100,
                'before_role_drift_percent': (medians['A2'] / medians['A1'] - 1) * 100,
                'after_role_drift_percent': (medians['B3'] / medians['B0'] - 1) * 100,
                'cross_role_output_hash_equal': len(hashes) == 1, 'slot_medians_ms': medians})
        records.append({'family': family, 'name': name, 'rounds': rows,
            'median_three_round_delta_percent': statistics.median(row['after_delta_percent'] for row in rows),
            'needs_main_regression_retest': any(row['after_delta_percent'] > 5 for row in rows),
            'max_absolute_role_drift_percent': max(abs(row[field]) for row in rows for field in ('before_role_drift_percent', 'after_role_drift_percent'))})
    report = {'comparison': args.comparison, 'actual_rounds': 3, 'actual_BAAB_runs': 12,
        'all89_shared_inputs_capture_DOT_samples_own_golden_wholebits_checked': True,
        'source_port_cross_role_wholebits_required': args.comparison == 'source-port',
        'case_columns': {'semantic_model51': [row for row in records if row['family'] == 'model51'],
                         'representative_direct38': [row for row in records if row['family'] == 'representative38']},
        'over5_percent_cases_requiring_new_main_retest_grant': [row['name'] for row in records if row['needs_main_regression_retest']],
        'input_raw_SHA': inputs, 'shared_run_identity': identity,
        'scope': 'not full model/end-to-end performance; semantic51 inclusive adapter graph and direct38 separate',
        'verdict': 'raw performance report only; main must review drift/confounds/positive deltas and authorize any retest'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

import argparse
import math
from pathlib import Path
import statistics

from protocol import METHODS, REPEATS, SAMPLES, code_hashes, derived, digest, graph_proof, load_plan, method_order, phase_plan, read, require, save, summarize


def audit_witnesses(root, cases):
    import torch

    require(not torch.xpu.is_initialized(), 'Independent full witness audit is CPU only')
    torch.set_num_threads(1)
    from runner import bits_sha, local, validate_output

    audited = {}
    for case in cases:
        require(digest(local(case['fixture']['path'])) == case['fixture']['sha256'], 'Independent original fixture SHA mismatch')
        require(digest(local(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256'], 'Independent expected SOURCE SHA mismatch')
        inputs = torch.load(local(case['fixture']['path']), map_location='cpu', weights_only=True)
        expected = torch.load(local(case['expected_SOURCE_output']), map_location='cpu', weights_only=True)['actual']
        expected_inputs = {'fixture_sha256': case['fixture']['sha256'], 'complete_QKVO_uint16_sha256': bits_sha(inputs['original']),
            'positions_dtype': str(inputs['positions'].dtype), 'complete_positions_bytes_sha256': bits_sha(inputs['positions']),
            'complete_GPU_BF16_weight_uint16_sha256': {name: bits_sha(inputs[name].to(torch.bfloat16)) for name in ('query_weight', 'key_weight')},
            'actual_GPU_input_full_bits_checked': True}
        for entry in phase_plan(cases):
            require(read(root / 'phases' / entry['phase'] / f"{case['name']}-actual-inputs.json") == expected_inputs, 'Actual full GPU input hashes differ from immutable complete fixture')
        outputs = {}
        for role in ('source', 'port'):
            for method in METHODS:
                path = root / 'witness' / role / case['name'] / f'{method}.pt'
                witness = torch.load(path, map_location='cpu', weights_only=True)
                for name in ('original', 'positions', 'query_weight', 'key_weight'):
                    require(torch.equal(witness[name], inputs[name]), 'All complete witness input tensors must equal original shared fixture')
                audited[f"{role}/{case['name']}/{method}"] = {'witness_sha256': digest(path),
                    'independent_full_CPUref_and_bits': validate_output(inputs['original'], witness['actual'], inputs, expected, method)}
                outputs[role, method] = witness['actual']
        for method in METHODS:
            require(torch.equal(outputs['source', method].view(torch.uint16), outputs['port', method].view(torch.uint16)), 'Full SOURCE/PORT actual witness bits mismatch')
        del inputs, expected, witness, outputs
    require(not torch.xpu.is_initialized(), 'Witness audit cannot initialize XPU')
    return audited


def analyze(root):
    plan, fields, cases = load_plan()
    grant = read(root / 'grant-copy.json')
    supervisor = read(root / 'supervisor.json')
    require(grant['runner_files_sha256'] == supervisor['runner_files_sha256'] == code_hashes(), 'Frozen actual driver changed')
    require(grant['providers'] == plan['providers'] and grant['all_fixture_sha256'] == {case['name']: case['fixture']['sha256'] for case in cases}, 'Provider/shared full input bindings changed')
    require(digest(root / 'grant-copy.json') == supervisor['external_grant_sha256'], 'Exact external main grant copy mismatch')
    from runner import local

    expected_stack = {str(local(path).resolve()): checksum for path, checksum in grant['expected_core_loaded_stack_sha256'].items()}
    require(read(root / 'phase-plan.json') == phase_plan(cases), 'Exactly three all19 BAAB rounds; no selected-case diagnostics')
    per_case = {case['name']: {method: {'source': [], 'port': [], 'rounds': []} for method in METHODS} for case in cases}
    raw_files = {}
    for entry in phase_plan(cases):
        require((root / f"{entry['phase']}.exit").read_text().strip() == '0', 'Actual phase failure preserved, not acceptance')
        directory = root / 'phases' / entry['phase']
        for case_index, name in enumerate(entry['cases']):
            case = next(item for item in cases if item['name'] == name)
            result = read(directory / f'{name}.json')
            require(result['case'] == name and result['phase'] == entry['phase'] and result['role'] == entry['role'], 'Case/provider order mismatch')
            require(result['fixture_sha256'] == case['fixture']['sha256'], 'Shared original input SHA mismatch')
            require(result['samples'] == SAMPLES and result['calls_per_sample'] == REPEATS, 'Fixed21x32 protocol')
            require(result['sample_method_orders'] == [list(method_order(entry['round'], entry['slot'], case_index, sample)) for sample in range(SAMPLES)], 'Recorded actual counterbalance order mismatch')
            require(len(result['per_call_raw_repeat_us']) == SAMPLES, 'Full per-call event arrays missing')
            for sample, values in enumerate(result['per_call_raw_repeat_us']):
                require(len(values) == REPEATS and all(math.isfinite(item) and item > 0 for item in values), 'Invalid or dropped individual per-call event')
                require(math.isclose(statistics.mean(values), result['methods'][METHODS[0]]['raw_us'][sample], rel_tol=1e-12), 'Per-call raw mean mismatch')
            for method in METHODS:
                samples = result['methods'][method]['raw_us']
                require(result['methods'][method] == summarize(samples), 'Independent median/MAD mismatch')
                proof = result['correctness'][method]
                require(proof['full_QKVO_bitwise'] and proof['tail_bitwise'], 'Actual full output bits not verified')
                if method != METHODS[2]:
                    require(proof['violations'] == 0 and proof['atol'] == proof['rtol'] == 0.005, 'Strict full CPUref5e-3 required')
                if 'full_witness' in proof:
                    require(digest(Path(proof['full_witness']['path'])) == proof['full_witness']['sha256'], 'Current full witness SHA mismatch')
                if method in METHODS[1:]:
                    totals = result['graph_raw_total_us'][method]
                    require(len(totals) == SAMPLES and all(math.isclose(total / REPEATS, sample, rel_tol=1e-12) for total, sample in zip(totals, samples)), 'Captured32-call total/raw mismatch')
                    proof_directory = directory / name / method
                    recorded = read(proof_directory / 'proof.json')
                    for filename, checksum in recorded['debug_DOT_sha256'].items():
                        require(digest(proof_directory / filename) == checksum, 'Real captured DOT changed')
                    trace = proof_directory / 'actual-replay.trace.json'
                    require(digest(trace) == recorded['actual_replay_trace_sha256'], 'Actual replay kernel trace changed')
                    validated = graph_proof('\n'.join((proof_directory / filename).read_text() for filename in recorded['debug_DOT_sha256']), read(trace), method)
                    require(all(recorded[key] == value for key, value in validated.items()), 'Backend graph/real kernel proof mismatch')
                    require(recorded['actual_captured_logical_calls'] == REPEATS and recorded['raw_backend_graph'] and recorded['raw_backend_exec'], 'Real graph required, not empty wrapper')
                row = per_case[name][method]
                row[entry['role']].extend(samples)
                row['rounds'].append({'round': entry['round'], 'slot': entry['slot'], 'role': entry['role'], **summarize(samples)})
            require(result['difference_diagnostic_only'] == derived(result['methods'][METHODS[1]]['raw_us'], result['methods'][METHODS[2]]['raw_us']), 'Diagnostic subtraction must preserve all negatives, not production kernel time')
        before = read(directory / 'runtime-start.json')
        after = read(directory / 'runtime-end.json')
        require(before['normal']['mapped_native_SHA'] == after['normal']['mapped_native_SHA'], 'Native maps changed')
        require(before['core_loaded_stack'] == after['core_loaded_stack'], 'Core loaded stack changed')
        require(before['core_loaded_stack'] == expected_stack, 'Actual loaded core stack differs from fresh main grant')
        for path in directory.rglob('*'):
            if path.is_file():
                raw_files[str(path.relative_to(root))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    for methods in per_case.values():
        for column in methods.values():
            for role in ('source', 'port'):
                values = column[role]
                require(len(values) == SAMPLES * 6, 'Every method/provider must retain allsix phases')
                median = statistics.median(values)
                column[f'{role}_summary'] = {'raw_us': values, 'median_us': median, 'MAD_us': statistics.median(abs(item - median) for item in values)}
            column['PORT_change_percent_same_method_only'] = (column['port_summary']['median_us'] / column['source_summary']['median_us'] - 1) * 100
            column['BAAB_drift_percent_by_round'] = []
            column['first_to_last_phase_drift_percent'] = {}
            for role in ('source', 'port'):
                rows = [row for row in column['rounds'] if row['role'] == role]
                column['first_to_last_phase_drift_percent'][role] = (rows[-1]['median_us'] / rows[0]['median_us'] - 1) * 100
            for round_index in range(3):
                values = [row for row in column['rounds'] if row['round'] == round_index]
                for role in ('source', 'port'):
                    pair = [row for row in values if row['role'] == role]
                    column['BAAB_drift_percent_by_round'].append({'round': round_index, 'role': role,
                        'percent': (pair[1]['median_us'] / pair[0]['median_us'] - 1) * 100})
    witnesses = audit_witnesses(root, cases)
    for path in root.rglob('*'):
        if path.is_file() and path.name not in ('analysis.json', 'analysis.log', 'analysis.exit'):
            raw_files[str(path.relative_to(root))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    save(root / 'analysis.json', {'actual_analysis_exit': 0, 'all19_cases': per_case, 'reviewed_plan_sha256': fields['plan_SHA'],
        'independent_full_CPU_output_witness_audit': witnesses,
        'raw_files_sha256': raw_files, 'methods_are_separate_not_cross_method_speedup': True,
        'difference_is_diagnostic_only_not_production_kernel_time': True, 'old_graph_results_NOT_reused': True,
        'old_per_call_regression_NOT_overridden_by_new_graph_column': True, 'all_performance_automatic_pass_claim': False})
    print('Actual all19/three BAAB data, graph nodes/kernels/ref/bits and independent raw statistics verified; no automatic performance PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve())

import argparse
import math
from pathlib import Path
import statistics

from protocol import METHODS, REPEATS, SAMPLES, code_hashes, derived, digest, dot_structure, replay_proof, load_plan, method_order, phase_plan, read, require, save, summarize


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


def graph_pair_audit(root, entry, case, method, timing, timing_directory):
    import torch
    from runner import local

    proof_directory = root / 'proof-phases' / entry['phase'] / case['name'] / method
    proof = read(proof_directory / 'proof.json')
    require(digest(proof_directory / 'proof.json') == timing['proofgraph_sha256'], 'Independent proofgraph binding')
    require(proof['pid'] != timing['pid'] and proof['profiler_exited_proofgraph_reset_synced'], 'Profiler process separation required')
    require(timing['fresh_unprofiled_timinggraph'] and timing['timinggraph_device_trace_NOT_collected'], 'Do not fake timinggraph device trace')
    for state in ('profiler_before', 'profiler_after'):
        require(timing[state]['native_profiler_enabled'] is False and timing[state]['Python_profiler_enabled'] is False
            and timing[state]['inherited_tracing_switches'] == {}, 'Inactive instrumentation required')
    for name in ('phase', 'role', 'fixture_sha256', 'native_SHA', 'core_SHA'):
        require(proof['bindings'][name] == timing['bindings'][name], 'Two graphs provider/core/fixture binding')
    for directory, recorded in ((proof_directory, proof), (timing_directory, timing)):
        require(recorded['raw_backend_graph'] and recorded['raw_backend_exec'] and recorded['actual_captured_logical_calls'] == REPEATS, 'Real distinct PID-scoped graph/exec')
        require(all(recorded['bindings']['input_pointers'].values()), 'Actual complete input GPU pointer bindings')
        for filename, checksum in recorded['debug_DOT_sha256'].items():
            require(digest(directory / filename) == checksum, 'Own actual DOT changed')
        dot = '\n'.join((directory / filename).read_text() for filename in recorded['debug_DOT_sha256'])
        require(all(recorded[key] == value for key, value in dot_structure(dot, method).items()), 'Actual DOT API classification')
        output = read(directory / 'whole-output.json')
        require(digest(directory / 'full-output.pt') == output['complete_witness_sha256'], 'Complete graph output witness SHA')
        inputs = torch.load(local(case['fixture']['path']), map_location='cpu', weights_only=True, mmap=True)
        witness = torch.load(directory / 'full-output.pt', map_location='cpu', weights_only=True, mmap=True)
        for name, tensor in inputs.items():
            require(torch.equal(tensor.view(torch.uint8), witness[name].view(torch.uint8)), 'Complete graph original inputs bitwise')
        expected = inputs['original'] if method == METHODS[2] else torch.load(local(case['expected_SOURCE_output']), map_location='cpu', weights_only=True, mmap=True)['actual']
        require(torch.equal(witness['actual'].view(torch.uint16), expected.view(torch.uint16)), 'Whole graph golden/reset bits')
        sentinel = torch.load(directory / 'sentinel.pt', map_location='cpu', weights_only=True, mmap=True)
        require(sentinel.shape == inputs['original'].shape and torch.isnan(sentinel).all().item(), 'Full sentinel proof')
        require(digest(directory / 'sentinel.pt') == read(directory / 'sentinel.json')['file_sha256'], 'Full sentinel file SHA')
    trace = proof_directory / 'full-window.trace.json'
    require(digest(trace) == proof['actual_replay_trace_sha256'], 'Original actual full-window trace SHA')
    dot = '\n'.join((proof_directory / name).read_text() for name in proof['debug_DOT_sha256'])
    validated = replay_proof(dot, read(trace), method)
    require(all(proof[key] == value for key, value in validated.items()), 'Original strict replay-specific device trace')


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
        require((root / f"{entry['phase']}-proof.exit").read_text().strip() == '0', 'Actual proof process terminal0')
        directory = root / 'phases' / entry['phase']
        proof_phase_directory = root / 'proof-phases' / entry['phase']
        proof_start = read(proof_phase_directory / 'runtime-start.json')
        proof_end = read(proof_phase_directory / 'runtime-end.json')
        timing_start = read(directory / 'runtime-start.json')
        require(proof_start['normal']['mapped_native_SHA'] == proof_end['normal']['mapped_native_SHA'] == timing_start['normal']['mapped_native_SHA'], 'Actual full native maps for both processes')
        require(proof_start['core_loaded_stack'] == proof_end['core_loaded_stack'] == timing_start['core_loaded_stack'], 'Actual shared core for both processes')
        isolation = read(directory / 'timing-process-isolation.json')
        require(isolation['proof_child_absent'] and isolation['timing_PID'] != isolation['proof_child_PID'], 'Actual process isolation record')
        for path in proof_phase_directory.rglob('*'):
            if path.is_file():
                raw_files[str(path.relative_to(root))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
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
                    graph_pair_audit(root, entry, case, method, recorded, proof_directory)
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

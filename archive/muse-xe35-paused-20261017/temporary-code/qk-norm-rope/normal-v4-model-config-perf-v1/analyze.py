import argparse
import json
import math
import statistics
from pathlib import Path


RUN = Path(__file__).resolve().parent


def summary(samples):
    median = statistics.median(samples)
    return {'median_us': median, 'MAD_us': statistics.median(abs(item - median) for item in samples),
        'sample_count': len(samples), 'minimum_us': min(samples), 'maximum_us': max(samples)}


def percentage(after, before):
    return (after / before - 1) * 100


def summarize_rounds(rounds):
    source = [sample for row in rounds for sample in row['source_raw_us']]
    port = [sample for row in rounds for sample in row['port_raw_us']]
    result = {'SOURCE': summary(source), 'PORT': summary(port), 'rounds': rounds}
    result['PORT_regression_percent'] = percentage(result['PORT']['median_us'], result['SOURCE']['median_us'])
    result['maximum_within_BAAB_absolute_drift_percent'] = max(
        abs(row[f'{role}_pair_drift_percent']) for row in rounds for role in ('source', 'port'))
    result['first_last_round_drift_percent'] = {role: percentage(
        rounds[-1][role]['median_us'], rounds[0][role]['median_us']) for role in ('source', 'port')}
    result['bounded_retest_needed'] = result['PORT_regression_percent'] > 5 or result['maximum_within_BAAB_absolute_drift_percent'] > 5 or any(
        abs(item) > 5 for item in result['first_last_round_drift_percent'].values())
    return result


def validate_result(result):
    assert result['samples'] == 21 and result['repeats'] == 32 and result['warmup'] == 5
    assert result['device_event'] and not result['graph'] and result['full_input_reset_each_call_outside_event']
    assert result['current_full_output_SOURCE_bitwise_sanity'] and result['current_tail_bitwise_sanity']
    assert len(result['raw_us']) == 21 and len(result['raw_per_repeat_us']) == 21
    for sample, repeats in zip(result['raw_us'], result['raw_per_repeat_us']):
        assert len(repeats) == 32 and all(math.isfinite(item) and item > 0 for item in repeats)
        assert math.isclose(sample, statistics.mean(repeats), rel_tol=1e-12)
    computed = summary(result['raw_us'])
    for key in ('median_us', 'MAD_us'):
        assert math.isclose(result[key], computed[key], rel_tol=1e-12, abs_tol=1e-12)


def analyze(stage):
    from run_perf import ORDER, data, digest, save, verify

    value = verify()
    plan = json.loads((RUN / 'phase-plan.json').read_text())
    names = [case['name'] for case in data()['cases']]
    initial = json.loads((RUN / 'analysis-initial.json').read_text()) if stage == 'final' else None
    selected = initial['bounded_retest_cases'] if initial else []
    rounds = range(6) if stage == 'final' and selected else range(3)
    assert len(plan) == len(rounds) * 4
    records = {}
    raw_files = {}
    for round_index in rounds:
        wanted = names if round_index < 3 else selected
        ordered = list(reversed(wanted)) if round_index % 2 else list(wanted)
        for slot, role in enumerate(ORDER):
            phase_id = f'r{round_index:02d}-s{slot}-{role}'
            entry = plan[phase_id]
            assert entry == {'round': round_index, 'slot': slot, 'role': role, 'cases': ordered, 'bounded_retest': round_index >= 3}
            directory = RUN / 'phases' / phase_id
            assert (directory / 'supervisor.exit').read_text().strip() == '0'
            runtime = json.loads((directory / 'runtime-audit.json').read_text())
            assert runtime['role'] == role and runtime['case_count'] == len(ordered)
            assert runtime['before']['mapped_native_SHA'] == runtime['after']['mapped_native_SHA']
            assert runtime['after']['mapped_count'] == (712 if role == 'port' else 155)
            for name in ordered:
                path = directory / f'{name}.json'
                result = json.loads(path.read_text())
                validate_result(result)
                case = next(item for item in value['cases'] if item['name'] == name)
                assert result['case'] == name and result['phase'] == phase_id and result['role'] == role
                assert result['fixture_sha256'] == case['fixture']['sha256'] and result['group'] == case['group']
                records[round_index, slot, name] = result
            for path in directory.iterdir():
                if path.is_file():
                    raw_files[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    cases = {}
    suspicious = []
    for case in value['cases']:
        name = case['name']
        rows = []
        for round_index in rounds:
            if round_index >= 3 and name not in selected:
                continue
            phases = [records[round_index, slot, name] for slot in range(4)]
            source = phases[1]['raw_us'] + phases[2]['raw_us']
            port = phases[0]['raw_us'] + phases[3]['raw_us']
            rows.append({'round': round_index, 'source': summary(source), 'port': summary(port),
                'source_raw_us': source, 'port_raw_us': port,
                'source_pair_drift_percent': percentage(phases[2]['median_us'], phases[1]['median_us']),
                'port_pair_drift_percent': percentage(phases[3]['median_us'], phases[0]['median_us']),
                'phase_medians_us': [phase['median_us'] for phase in phases]})
        first = summarize_rounds(rows[:3])
        if first['bounded_retest_needed']:
            suspicious.append(name)
        retry = summarize_rounds(rows[3:]) if len(rows) > 3 else None
        decisive = retry or first
        status = 'regression_confirmed' if decisive['PORT_regression_percent'] > 5 else (
            'drift_requires_qualification' if decisive['bounded_retest_needed'] else 'within_5_percent_nonregression')
        cases[name] = {'group': case['group'], 'initial': first, 'bounded_retest': retry,
            'acceptance_status': status, 'comparison': 'optimized SOURCE to PORT migration fidelity; not preopt optimization speedup'}
    if initial:
        assert suspicious == selected
    output = {'actual_analysis_exit': 0, 'version': value['version'], 'stage': stage, 'case_count': len(cases),
        'model_config_semantic_synthetic_cases': 17, 'legacy_representative_cases': 2,
        'bounded_retest_cases': suspicious, 'cases': cases, 'raw_files': raw_files,
        'eight_card_concurrency_claim': False, 'actual_raw_model_replay': False,
        'failed_preopt3508_used_as_correctness_matched_baseline': False,
        'all_cases_unqualified_performance_nonregression': all(case['acceptance_status'] == 'within_5_percent_nonregression' for case in cases.values())}
    save(RUN / f'analysis-{stage}.json', output)
    print(json.dumps({key: item for key, item in output.items() if key not in ('cases', 'raw_files')}), flush=True)


def self_test():
    row = {'source_raw_us': [10.0] * 42, 'port_raw_us': [10.4] * 42,
        'source': summary([10.0] * 42), 'port': summary([10.4] * 42),
        'source_pair_drift_percent': 0.0, 'port_pair_drift_percent': 0.0}
    assert not summarize_rounds([row] * 3)['bounded_retest_needed']
    changed = {**row, 'port_raw_us': [10.6] * 42}
    assert summarize_rounds([changed] * 3)['bounded_retest_needed']
    assert summarize_rounds([{**row, 'port_pair_drift_percent': 6.0}] * 3)['bounded_retest_needed']
    result = {'samples': 21, 'repeats': 32, 'warmup': 5, 'device_event': True, 'graph': False,
        'full_input_reset_each_call_outside_event': True, 'current_full_output_SOURCE_bitwise_sanity': True,
        'current_tail_bitwise_sanity': True, 'raw_us': [10.0] * 21, 'raw_per_repeat_us': [[10.0] * 32] * 21,
        'median_us': 10.0, 'MAD_us': 0.0}
    validate_result(result)
    try:
        validate_result({**result, 'raw_per_repeat_us': [[0.0] * 32] * 21})
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid timing must be rejected')
    print('CPU-only analyzer self-test actual0; no GPU or timing executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('initial', 'final'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    assert bool(args.stage) != args.self_test
    self_test() if args.self_test else analyze(args.stage)

import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
from run_perf import ACCURACY, ARTIFACT, RUN, digest, local_path, save, verify


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    value = verify()
    for name in ('prepare', 'cpu-preflight', 'grant-publication', 'analysis-initial', 'analysis-final', 'host-supervisor'):
        assert (RUN / f'{name}.exit').read_text().strip() == '0'
    grant = json.loads((RUN / 'performance-grant.json').read_text())
    assert grant['gpu'] == 6 and grant['driver_sha256'] == value['driver_sha256']
    assert not grant['eight_card_concurrency_claim_allowed'] and not grant['W4_lease_transfer_authorized']
    final = json.loads((RUN / 'analysis-final.json').read_text())
    assert final['actual_analysis_exit'] == 0 and final['case_count'] == 19
    for name, expected in final['raw_files'].items():
        path = RUN / name
        assert digest(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
    for case in value['cases']:
        assert digest(local_path(case['fixture']['path'])) == case['fixture']['sha256']
        assert digest(local_path(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256']
    baseline = {}
    for role in ('source', 'port'):
        cpu = json.loads((RUN / f'cpu-normal-{role}.json').read_text())
        assert cpu['actual_exit'] == 0 and not cpu['XPU_initialized'] and cpu['case_count'] == 19
        baseline[role] = cpu['maps']['mapped_native_SHA']
        assert len(baseline[role]) == (155 if role == 'source' else 712)
        for name, checksum in baseline[role].items():
            assert digest(local_path(name)) == checksum
    plan = json.loads((RUN / 'phase-plan.json').read_text())
    measured_calls = 0
    round_records = {}
    for phase_id, phase in plan.items():
        directory = RUN / 'phases' / phase_id
        assert (directory / 'supervisor.exit').read_text().strip() == '0'
        command = json.loads((directory / 'supervisor.command.json').read_text())
        assert command['cpu_affinity'] == list(range(32, 48)) and command['ZE_AFFINITY_MASK'] == '6'
        runtime = json.loads((directory / 'runtime-audit.json').read_text())
        for point in ('before', 'after'):
            assert runtime[point]['mapped_native_SHA'] == baseline[phase['role']]
        log = (directory / 'supervisor.log').read_text().splitlines()
        stdout = {json.loads(line)['case']: json.loads(line) for line in log if line.startswith('{') and '"case"' in line}
        assert set(stdout) == set(phase['cases'])
        for name in phase['cases']:
            result = json.loads((directory / f'{name}.json').read_text())
            assert stdout[name] == {key: item for key, item in result.items() if not key.startswith('raw_')}
            assert result['current_full_output_SOURCE_bitwise_sanity'] and result['current_tail_bitwise_sanity']
            assert len(result['raw_us']) == 21 and len(result['raw_per_repeat_us']) == 21
            for mean, calls in zip(result['raw_us'], result['raw_per_repeat_us']):
                assert len(calls) == 32 and all(math.isfinite(item) and item > 0 for item in calls)
                assert math.isclose(statistics.mean(calls), mean, rel_tol=1e-12)
                measured_calls += len(calls)
            median = statistics.median(result['raw_us'])
            assert math.isclose(median, result['median_us'], rel_tol=1e-12)
            assert math.isclose(statistics.median(abs(item - median) for item in result['raw_us']), result['MAD_us'], rel_tol=1e-12, abs_tol=1e-12)
            round_records[phase['round'], phase['slot'], name] = result
    for name, case in final['cases'].items():
        for bucket in ('initial', 'bounded_retest'):
            if case[bucket] is None:
                continue
            source = []
            port = []
            for row in case[bucket]['rounds']:
                phases = [round_records[row['round'], slot, name] for slot in range(4)]
                assert row['phase_medians_us'] == [phase['median_us'] for phase in phases]
                assert row['source_raw_us'] == phases[1]['raw_us'] + phases[2]['raw_us']
                assert row['port_raw_us'] == phases[0]['raw_us'] + phases[3]['raw_us']
                source.extend(row['source_raw_us'])
                port.extend(row['port_raw_us'])
            source_median = statistics.median(source)
            port_median = statistics.median(port)
            assert source_median == case[bucket]['SOURCE']['median_us'] and port_median == case[bucket]['PORT']['median_us']
            assert math.isclose((port_median / source_median - 1) * 100, case[bucket]['PORT_regression_percent'], rel_tol=1e-12, abs_tol=1e-12)
    frozen = json.loads((ARTIFACT / 'canonical-source-freeze-v4-icpx.json').read_text())
    target = ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35'
    for name, expected in frozen['files'].items():
        assert digest(target / name) == expected['sha256']
    preopt = ARTIFACT / 'source-accuracy-v1/fresh-preopt/accuracy-t32768'
    assert preopt.with_suffix('.exit').read_text().strip() == '1'
    assert json.loads(preopt.with_suffix('.json').read_text())['violations'] == 3508
    observed = [int((RUN / 'host-supervisor.pid').read_text())]
    observed.extend(json.loads(path.read_text())['pid'] for path in RUN.rglob('*.pid.json'))
    observed.append(json.loads((RUN / 'gpu-supervisor.json').read_text())['pid'])
    continued = json.loads((RUN / 'actual-CONT-proof.json').read_text())
    assert continued['actual_exit'] == 0 and continued['same_process_starttick'] == 8042282
    assert continued['main_release_sha256'] == digest(RUN / 'main-continue-release.json')
    release = json.loads((RUN / 'main-continue-release.json').read_text())
    assert digest(RUN / 'main-drain-evidence-snapshot.json') == release['drain_evidence_sha256']
    assert release['old_immutable_grant_sha256'] == digest(RUN / 'performance-grant.json')
    assert release['frozen_driver_files'] == json.loads((RUN / 'driver-freeze.json').read_text())['files']
    assert release['process_starttick'] == 8042282 and release['verified_state_before_CONT'] == 'T'
    observed.append(continued['host_PID'])
    live = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = (directory / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
            comm = (directory / 'comm').read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm.startswith('python') and 'normal-v4-model-config-perf-v1/run_perf.py' in arguments:
            live.append({'pid': int(directory.name), 'cmdline': arguments})
        if int(directory.name) in observed and 'normal-v4-model-config-perf-v1' in arguments:
            live.append({'pid': int(directory.name), 'cmdline': arguments})
    assert not live, f'Own SOURCE/PORT runtime not drained: {live}'
    lock = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], capture_output=True, text=True)
    assert lock.returncode == 0, 'GPU6 physical lock still held; do not claim release'
    save(RUN / 'drain-check.json', {'observed_own_PIDs': sorted(set(observed)), 'matching_live_own_processes': live,
        'drained': True, 'GPU6_physical_lock_probe_exit': lock.returncode, 'logical_GPU6_transfer_to_W4': False})
    raw = {}
    excluded = {'final-evidence-audit.json', 'final-evidence-audit.log', 'final-evidence-audit.exit', 'write-set.json'}
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.suffix != '.md' and path.name not in excluded:
            raw[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    save(RUN / 'final-evidence-audit.json', {'actual_exit': 0, 'CPU_auditor_pid': os.getpid(),
        'driver_sha256': value['driver_sha256'], 'grant_sha256': digest(RUN / 'performance-grant.json'),
        'case_count': 19, 'model_config_semantic_synthetic_count': 17, 'legacy_representative_count': 2,
        'phase_count': len(plan), 'measured_event_call_count': measured_calls, 'raw_files': raw,
        'SOURCE_mapped_native_count': 155, 'PORT_mapped_native_count': 712,
        'every_phase_current_full_QKV_and_tail_SOURCE_bitwise_sanity': True,
        'all_raw_event_samples_median_MAD_recomputed': True, 'all_shared_input_SHA_unchanged': True,
        'old_references_frozen_source_API_generic_BMG_math_unchanged': True,
        'preopt3508_actual_exit1_preserved_not_used_as_matched_timing_baseline': True,
        'analysis_final_sha256': digest(RUN / 'analysis-final.json'),
        'unqualified_nonregression': final['all_cases_unqualified_performance_nonregression'],
        'GPU_processes_drained': True, 'physical_GPU6_lock_released': True,
        'logical_GPU6_lease_transferred_to_W4': False, 'eight_card_concurrent_launch_claim': False,
        'raw_model_replay_claim': False, 'all_production_invocations_verified': False,
        'prior_complete_accuracy_audit_sha256': digest(ACCURACY / 'final-evidence-audit.json')})
    print(json.dumps({'actual_exit': 0, 'cases': 19, 'phases': len(plan), 'measured_event_calls': measured_calls,
        'raw_files': len(raw), 'physical_GPU6_lock_released': True,
        'unqualified_nonregression': final['all_cases_unqualified_performance_nonregression']}, indent=2))


if __name__ == '__main__':
    main()

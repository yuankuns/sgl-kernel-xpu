import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    artifact = Path(__file__).resolve().parent
    runtime = artifact / 'official-performance-v4-gpu6-v2'
    output = artifact / 'official-performance-v4-gpu6-v2-post'
    phases = []
    for phase, role in enumerate(['source', 'after', 'after', 'source'] * 3):
        assert int((runtime / f'phase-{phase}-{role}.exit').read_text()) == 0
        phases.append(json.loads((runtime / f'phase-{phase}-{role}.json').read_text()))
    assert int((runtime / 'host-launch.exit').read_text()) == int((runtime / 'gpu-runtime.exit').read_text()) == int((runtime / 'performance-analysis.exit').read_text()) == 5
    assert all(len(records) == 5 for records in phases)
    assert not (runtime / 'performance-maintenance-hold.json').exists()
    baseline = json.loads((artifact / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    for records in phases:
        for actual, expected in zip(records, baseline):
            assert actual['shape'] == expected['shape']
            assert actual['input_hashes'] == expected['inputs']
            assert actual['full_valid_output_hash'] == expected['full_valid_output']
            assert actual['genuine_replay_NaN_sentinel_full_valid_output_proof']
            assert actual['unwritten_slack_rows_excluded'] == (1 if actual['shape'].startswith('257-') else 0)
            assert len(actual['raw_per_call_ms']) == 31
    for role, keys in [('source', ['package_files_sha256', 'mapped_native_sha256', 'provenance_sha256', 'Torch_provider', 'Torch_version']),
                       ('after', ['manifest_sha256', 'mapped_native_hashes', 'source_digest', 'compiler_policy'])]:
        assert int((output / f'CPU-normal-{role}.exit').read_text()) == 0
        before = json.loads((runtime / f'CPU-normal-{role}.json').read_text())
        after = json.loads((output / f'CPU-normal-{role}.json').read_text())
        assert all(before[key] == after[key] for key in keys)
    owners = [int((runtime / name).read_text()) for name in ['host-waiter.pid', 'host-supervisor.pid', 'maintenance-monitor.pid']]
    assert all(not Path('/proc', str(pid)).exists() for pid in owners)
    with Path('/tmp/intel-gpu-6.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    container_proof = artifact.parent / 'routing-quant/fresh-current-source-native-v3-preparation/root-readonly-stage-protection-and-container-drain.json'
    container = json.loads(container_proof.read_text())
    assert int((runtime / 'container-runtime.pid').read_text()) in container['old_known_container_runtime_PIDs_absent']
    assert not container['active_nonZ_X_GPU4_6_7_environment_processes']
    analysis = json.loads((runtime / 'performance-analysis.json').read_text())
    rows = []
    for index, shape in enumerate(analysis['shapes']):
        for cycle in range(3):
            roles = phases[cycle * 4:cycle * 4 + 4]
            row = {'shape': shape['shape'], 'cycle': cycle,
                   'B0_SOURCE_ms': roles[0][index]['median_ms'], 'A0_AFTER_ms': roles[1][index]['median_ms'],
                   'A1_AFTER_ms': roles[2][index]['median_ms'], 'B1_SOURCE_ms': roles[3][index]['median_ms'],
                   'cycle_change_pct': shape['three_BAAB_cycle_changes_pct'][cycle],
                   **shape['within_cycle_drift'][cycle]}
            rows.append(row)
    with (output / 'all-five-shapes-three-BAAB-role-medians.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_files = [path for path in runtime.iterdir() if path.is_file() and path.suffix != '.pt']
    report = {'epoch': time.time(), 'CPU_terminal_evidence_actual_exit': 0, 'true_exec_supervisor_handle': 1695,
              'true_exec_supervisor_exit': 5, 'true_host_GPU_analysis_exits': [5, 5, 5],
              'all12_role_phase_exits': [0] * 12, 'all60_shape_role_full_valid_SOURCE_bits_verified': True,
              'shared_input_scale_rawSHA_verified': True, 'partial_FP64_scope': 'original max_rows4/expert/fullcolumns; not every production row oracle',
              'prior_SOURCE_both_providers_and_normal_AFTER_original40_precision_pass_preserved': True,
              'normal_SOURCE_AFTER_provider_protected_source_native_flags_post_unchanged': True,
              'known_host_owner_PIDs_absent': owners, 'container_runtime_PID_absent': int((runtime / 'container-runtime.pid').read_text()),
              'container_physical_drain_proof_SHA256': digest(container_proof), 'GPU6_physical_lock_free': True,
              'no_maintenance_hold_or_external_intervention': True, 'performance_gate_passed': False,
              'no_new_retake_math_launch_dispatch_changes': True, 'shapes': analysis['shapes'], 'all_BAAB_role_medians': rows,
              'raw_evidence_SHA256': {path.name: digest(path) for path in raw_files}}
    (output / 'W4_TERMINAL_PRECISION_PERFORMANCE_DRAIN_v2.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ['shapes', 'all_BAAB_role_medians', 'raw_evidence_SHA256']}, indent=2))


if __name__ == '__main__':
    main()

import json
from pathlib import Path
import sys

from launch_monitor import digest


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
FORENSIC = ART / 'v5-capture-forensic-v2'
RUN = FORENSIC / 'future-run-v1'
sys.path.insert(0, str(ART / 'v5-gpu5-source-port-performance-v1'))
from host_scan import snapshot


def main():
    freeze = json.loads((FORENSIC / 'CPU_READY_FINAL_SEAL_v2.json').read_text())
    for name, checksum in freeze['file_SHA'].items():
        assert digest(FORENSIC / name) == checksum, name
    monitor = json.loads((ROOT / 'resource-monitor-terminal.json').read_text())
    closed = json.loads((ROOT / 'grant-closed.json').read_text())
    pair = [json.loads(line) for line in (RUN / 'pair-protocol-markers.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (RUN / 'source/protocol-markers.jsonl').read_text().splitlines()]
    report = json.loads((RUN / 'source.json').read_text())
    assert int((RUN / 'source.exit').read_text()) == pair[-1]['actual_returncode'] == -11
    assert monitor['actual_suite_exit'] == int((ROOT / 'suite.exit').read_text()) == 139
    assert monitor['guard'] is None and monitor['remaining_known_live_descendants'] == {}
    assert closed['closed'] is True and closed['runtime_reuse_permitted'] is False
    assert not (RUN / 'port.json').exists() and not (RUN / 'port.log').exists()
    assert len(report['cases']) == 1 and report['cases'][0]['generation'] == 1
    assert report['cases'][0]['eager_and_all8_replay_SOURCE_wholebits'] is True
    assert len(report['cases'][0]['all_eight_replay_outputs']) == 8
    references = [row for row in rows if row['stage'] == 'full_reference_PASS']
    assert len(references) == 10
    total = 0
    for row in references:
        reference = row.get('reference', row.get('full_reference'))
        assert reference['reference_PASS'] and reference['coverage'] == 'full'
        assert reference['atol'] == reference['rtol'] == 0.005 and row['SOURCE_wholebits'] is True
        assert reference['checked_elements'] == 100663296
        total += reference['checked_elements']
    assert total == 1006632960
    assert rows[-1]['stage'] == 'before_capture_end' and rows[-1]['generation'] == 2
    for generation in (1, 2):
        observed = [row for row in rows if row['stage'] == 'actual_fwd_return' and row['generation'] == generation]
        assert len(observed) == 8 and all(row['observation']['native_returned'] for row in observed)
        assert all(row['observation']['scalars']['max_seqlen_k'] == 4096 for row in observed)
        assert all(row['observation']['scalars']['is_causal'] is True for row in observed)
    assert not any(row['stage'] == 'capture_end_returned' and row['generation'] == 2 for row in rows)
    assert any(row['stage'] == 'cleanup_returned' and row['generation'] == 1 for row in rows)
    for row in rows:
        if row['stage'].endswith('_saved'):
            path = Path(row['saved_path'].replace('/root/', '/home/syk/', 1))
            assert digest(path) == row['saved_file_sha256'], path.name
    assert digest(RUN / 'source/g1.dot') == report['cases'][0]['actual_DOT_sha256']
    assert not (RUN / 'source/g2.dot').exists()
    scan = snapshot()
    for pid, start in monitor['known_start_time_bound_host_PIDs'].items():
        path = Path(f'/proc/{pid}/stat')
        if path.exists():
            fields = path.read_text().rsplit(')', 1)[1].split()
            assert int(fields[19]) != start or fields[0] in ('Z', 'X')
    files = sorted(path for path in RUN.rglob('*') if path.is_file())
    files += [ROOT / name for name in (
        'gpu5-paired-forensic-barrier-v2.json', 'launch_monitor.py', 'supervisor-start.json', 'wrapper-start.json',
        'resource-monitor-terminal.json', 'after-suite-drain-lock.json', 'grant-closed.json',
        'admission-before-launch.json', 'suite.log', 'suite.exit')]
    sealed = {
        'CPU_terminal_seal_actual_exit': 0, 'actual_SOURCE_child_exit': -11, 'actual_suite_exit': 139,
        'classification': 'SOURCE generation2 SIGSEGV after all8 native calls returned; last marker before_capture_end; no core-derived exact native stack claim',
        'SOURCE_completed_generations': 1, 'SOURCE_gen1_all8_replay_fullrefs_wholebits': True,
        'SOURCE_gen2_eager_fullref_wholebits': True, 'completed_reference_evaluations': 10,
        'checked_reference_elements': total, 'SOURCE_actual_fwd_returns_each_generation': [8, 8],
        'PORT_started': False, 'gen2_capture_end_returned': False, 'gen2_DOT_available': False,
        'gen2_replay_outputs_saved': False, 'gen2_before_capture_full_inputs_and_eager_saved': True,
        'last_protocol_marker': rows[-1], 'own_scope_independent_drain_and_free_lock': scan,
        'v2_code_CPU_freeze_unchanged': True, 'automatic_retry': False, 'performance_acceptance': False,
        'shared_cause_with_W4_proven': False, 'GPU5_FM_logical_lease_retained': True, 'NV_handoff_authorized': False,
        'file_SHA256': {str(path.relative_to(ART)): digest(path) for path in files},
    }
    with (ROOT / 'TERMINAL_SEAL_CPU_v2.json').open('x') as stream:
        stream.write(json.dumps(sealed, indent=2) + '\n')
    print(json.dumps({key: sealed[key] for key in ('CPU_terminal_seal_actual_exit', 'actual_suite_exit',
        'SOURCE_completed_generations', 'checked_reference_elements', 'PORT_started', 'automatic_retry')}))


if __name__ == '__main__':
    main()

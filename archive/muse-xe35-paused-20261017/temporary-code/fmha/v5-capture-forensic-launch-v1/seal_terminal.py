import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
RUN = ART / 'v5-capture-forensic-v1/future-run-v1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    monitor = json.loads((ROOT / 'resource-monitor-terminal.json').read_text())
    drain = json.loads((ROOT / 'after-suite-drain-lock.json').read_text())
    closed = json.loads((ROOT / 'grant-closed.json').read_text())
    markers = [json.loads(line) for line in (RUN / 'source/protocol-markers.jsonl').read_text().splitlines()]
    references = [row for row in markers if row['stage'] == 'full_reference_PASS']
    schema = markers[0]['actual_registered_schema']
    assert 'bool is_causal' in schema and 'bool causal,' not in schema
    assert "forensic firstfail: KeyError('causal')" in (RUN / 'source.log').read_text()
    assert int((RUN / 'source.exit').read_text()) == monitor['actual_suite_exit'] == 2
    assert monitor['remaining_known_live_descendants'] == {} and monitor['guard'] is None
    assert drain['actual_exit'] == 0 and drain['GPU5_actual_flock_free'] is True
    assert closed['closed'] is True and closed['runtime_reuse_permitted'] is False
    assert not (RUN / 'port.json').exists() and not (RUN / 'port.log').exists()
    assert len(references) == 1 and references[0]['SOURCE_wholebits'] is True
    assert references[0]['reference']['checked_elements'] == 100663296
    assert references[0]['reference']['atol'] == references[0]['reference']['rtol'] == 0.005
    assert markers[-1]['stage'] == 'capture_launch_begin' and markers[-1]['call'] == 1
    assert not any(row['stage'] == 'actual_fwd_enter' for row in markers)
    files = sorted(path for path in RUN.rglob('*') if path.is_file())
    files += [ROOT / name for name in (
        'gpu5-paired-forensic-barrier-v1.json', 'launch_monitor.py', 'supervisor-start.json',
        'wrapper-start.json', 'resource-monitor-terminal.json', 'after-suite-drain-lock.json',
        'grant-closed.json', 'admission-before-launch.json', 'suite.log', 'suite.exit')]
    report = {
        'CPU_terminal_seal_actual_exit': 0,
        'suite_actual_exit': 2,
        'classification': 'forensic observer schema-key error before first observed native fwd; not a reproduced SIGSEGV',
        'schema': schema,
        'last_protocol_marker': markers[-1],
        'completed_full_reference_evaluations': len(references),
        'checked_reference_elements': sum(row['reference']['checked_elements'] for row in references),
        'SOURCE_eager_wholebits': True,
        'completed_capture_generations': 0,
        'PORT_started': False,
        'full89_or_performance_acceptance': False,
        'automatic_retry': False,
        'owned_descendants_drained': True,
        'GPU5_actual_flock_free': True,
        'GPU5_logical_FM_lease_retained': True,
        'NV_handoff_authorized': False,
        'application_disk_upper_bound_bytes': 32 * 1024**3,
        'monitor_terminal': monitor,
        'file_SHA256': {str(path.relative_to(ART)): digest(path) for path in files},
    }
    output = ROOT / 'TERMINAL_SEAL_CPU_v1.json'
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'CPU_terminal_seal_actual_exit', 'suite_actual_exit', 'checked_reference_elements',
        'owned_descendants_drained', 'GPU5_actual_flock_free', 'automatic_retry')}))


if __name__ == '__main__':
    main()

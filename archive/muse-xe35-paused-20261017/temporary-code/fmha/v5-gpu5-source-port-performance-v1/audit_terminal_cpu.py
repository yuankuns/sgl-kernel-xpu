import hashlib
import json
from pathlib import Path
import time

from host_scan import snapshot


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (ART / 'model89-normal-v5-gpu5-source-port-performance-v1.exit').read_text().strip() == '139'
    assert (ROOT / 'suite.exit').read_text().strip() == '139'
    assert (ROOT / 'source-port-r1-B0.exit').read_text().strip() == '-11'
    progress = json.loads((ROOT / 'suite-progress.json').read_text())
    assert len(progress['slots']) == 1 and progress['stopped_on_first_failure'] and not progress['auto_retry']
    raw = json.loads((ROOT / 'source-port-r1-B0.json').read_text())
    assert len(raw['cases']) == 1 and not raw.get('all89_input_SHA_capture_nodes_wholebits_samples_PASS', False)
    case = raw['cases'][0]
    assert case['wholebits_before_timing_PASS'] and case['wholebits_after_timing_PASS']
    assert len(case['samples_ms']) == 21 and all(sample > 0 for sample in case['samples_ms'])
    assert len(case['capture_actual_registered_fwd_nodes']) == 8
    graph = ROOT / 'source-port-r1-B0-graphs' / Path(case['actual_graph_debug_dump']).name
    assert digest(graph) == case['actual_graph_debug_dump_sha256']
    output = ROOT / 'source-port-r1-B0-full-outputs' / Path(case['saved_output_file']).name
    assert digest(output) == case['saved_output_file_sha256']
    assert digest(ART / 'v5-gpu5-performance-preparation/performance_driver.py') == 'c96e671fbc392e8d1b4f4c54346cd40a89b82468206b0b8832cef4ea6bf016e7'
    assert digest(ROOT / 'gpu5-source-port-barrier-v1.json') == '71bff1c53ac773c10382fc1f7841a0b84811b2db1db7ebf1d9d485dbec64856a'
    assert digest(ART / 'v5-model51-gpu10-preparation/normal-v5-gpu10-accuracy.json') == '90fbd175fbb525799e77df8e943714c7d5f3c4b20ad9c3d8b4c4c54cd19d1643'
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    actual = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35')
    assert len(candidate['files']) == 660
    for name, record in candidate['files'].items():
        assert digest(actual / name) == record['sha256'], name
    pids = (3301680, 3301681, 3302833, 3302838, 3302898, 3303047)
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    scan = snapshot()
    assert scan['GPU5_actual_flock_free']
    assert json.loads((ROOT / 'grant-closed.json').read_text())['closed']
    metadata = json.loads((ROOT / 'APPORT_MATCHED_PID_METADATA.json').read_text())
    assert metadata['matched_host_pid'] == 3303047 and metadata['fields']['Signal'] == '11'
    trace = (ROOT / 'core-backtrace-CPU-host-v3.log').read_text()
    for symbol in ('L0::CommandList::setKernelState', 'urGraphInstantiateGraphExp', 'capture_end'):
        assert symbol in trace
    assert (ROOT / 'core-backtrace-CPU-host-v3.exit').read_text().strip() == '0'
    report = {'epoch': time.time(), 'CPU_terminal_audit_actual_exit': 0,
        'actual_suite_exit': 139, 'actual_first_PORT_B0_child_returncode': -11,
        'completed_BAAB_rounds': 0, 'started_slots': 1, 'unstarted_slots': 11,
        'first_case': case['name'], 'first_case_actual_DOT_capture8_samples21_wholebits': True,
        'partial_samples_not_speedup_or_acceptance': True, 'raw_sha256': digest(ROOT / 'source-port-r1-B0.json'),
        'actual_precision89_prior_evidence_unchanged': True, 'frozen_actual_PR660_SHA_unchanged': True,
        'drained_host_PIDs': list(pids), 'actual_host_scan': scan,
        'grant_closed': True, 'immutable_grant_bytes_unchanged': True, 'no_retry': True,
        'GPU5_physical_flock_released': True, 'GPU5_FM_logical_lease_retained': True, 'NV_handoff_authorized': False,
        'GPU3_grant_issued': False, 'GPU3_preopt_requires_Kepler10_actual51_precision0_then_main_freshgrant': True,
        'verified_crash_stage': 'later XPUGraph capture_end/instantiate -> SYCL finalize -> UR graph instantiate -> L0 setKernelState cmdlist.cpp:284',
        'causal_boundary': 'host graph instantiation crash proven; exact kernel-handle ownership/lifetime bug and minimal repair not proven; not a precision failure or before/after regression claim',
        'core_scope': 'own private CPU-only extracted core files mode0600; excluded from source/PR payload; original system crash read-only',
        'new_GPU_or_compile_after_failure': False,
        'raw_evidence_SHA': {name: digest(ROOT / name) for name in ('source-port-r1-B0.log', 'source-port-r1-B0.exit',
            'suite.log', 'suite.exit', 'suite-progress.json', 'grant-closed.json', 'APPORT_MATCHED_PID_METADATA.json',
            'CORE_EXTRACTION_CPU.json', 'core-backtrace-CPU-host-v3.log', 'core-backtrace-CPU-host-v3.exit')}}
    with (ROOT / 'FAILED_SUITE_TERMINAL_CPU_AUDIT.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('CPU_terminal_audit_actual_exit', 'actual_suite_exit',
        'completed_BAAB_rounds', 'started_slots', 'unstarted_slots', 'GPU5_physical_flock_released', 'no_retry', 'raw_sha256')}))


if __name__ == '__main__':
    main()

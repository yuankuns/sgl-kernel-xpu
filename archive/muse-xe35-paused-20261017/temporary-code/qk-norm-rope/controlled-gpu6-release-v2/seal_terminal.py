import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'controlled-method-runner-cpu-v2'
sys.path.insert(0, str(V2))
from protocol import code_hashes, digest, read, require, save


def host():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light host drainCPU0–15')
    pids = [read(ROOT / 'actual-host-supervisor.json')['pid'], read(ROOT / 'host-launch.pid.json')['pid']]
    require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Own host process not absent')
    probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], check=False)
    require(probe.returncode == 0, 'PhysicalGPU6 still locked')
    save(ROOT / 'actual-host-final-drain.json', {'actual_exit': 0, 'epoch': time.time(),
        'owned_host_PIDs_absent': pids, 'physical_GPU6_lock_free_actual_exit': probe.returncode,
        'logical_GPU6_handoff_requires_main': True})
    print('Own host absent/physicalGPU6 free actual0', flush=True)


def seal():
    require(sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Full raw SHA sealCPU32–47')
    grant = read(ROOT / 'main-fresh-grant.json')
    run = V2 / 'runs' / grant['release_id']
    require(code_hashes() == grant['runner_files_sha256'], 'Frozen v2 code changed')
    require(digest(V2 / 'READY.json') == grant['cpu_READY_sha256'], 'Frozen READY changed')
    monitor = read(ROOT / 'resource-monitor-terminal.json')
    require(monitor['actual_exit'] == 0 and monitor['violation'] is None, 'Resource violation not expected')
    pids = [int(pid) for pid in monitor['known_PID_startticks']] + [read(ROOT / 'resource-monitor-ready.json')['pid']]
    require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Own container process not absent')
    status = read(ROOT / 'external-supervisor-terminal.json')
    require(status['actual_launch_exit'] == 1 and status['actual_monitor_exit'] == 0, 'Preserve actual failure1/monitor0')
    require(read(ROOT / 'actual-host-final-drain.json')['actual_exit'] == 0, 'Real host/drain0 required')
    require((run / 'r00-s0-port-proof.exit').read_text().strip() == '1', 'Actual first proof child1')
    require(not (run / 'phases').exists() and not list(run.rglob('raw-progress.json')), 'No timing samples permitted after firstfail')
    ledger = {str(path.relative_to(ROOT.parent)): {'sha256': digest(path), 'bytes': path.stat().st_size}
        for base in (run, ROOT, V2 / 'supervisors' / grant['release_id']) for path in sorted(base.rglob('*')) if path.is_file()}
    save(ROOT / 'TERMINAL.json', {'independent_terminal_audit_actual_exit': 0, 'epoch': time.time(),
        'actual_exec_handle': 37194, 'actual_runtime_exit': 1, 'actual_monitor_exit': 0,
        'first_failed_phase': 'r00-s0-port-proof', 'first_failed_case': 'prefill-wave-5req-20480',
        'first_failed_method': 'captured_reset_adapter', 'first_failure': 'Full QKVO uint16 output mismatch',
        'first12288_G_and_R_proof_completed': True, 'all19_suite_complete': False,
        'timing_child_started': False, 'performance_samples': 0, 'SOURCE_phase_started': False,
        'resource_violation': None, 'max_worker_RSS_bytes': monitor['max_worker_RSS_bytes'],
        'max_tree_RSS_bytes': monitor['max_tree_RSS_bytes'], 'owned_container_PIDs_absent': pids,
        'actual_host_drain_sha256': digest(ROOT / 'actual-host-final-drain.json'),
        'physical_GPU6_lock_free_actual_exit': 0, 'frozen_READY_sha256': digest(V2 / 'READY.json'),
        'fresh_grant_sha256': digest(ROOT / 'main-fresh-grant.json'),
        'all_actual_raw_commands_logs_exits_SHA256': ledger, 'no_retry_or_scope_math_tolerance_change': True,
        'old_v1_trace1_and_preopt3508_preserved': True, 'no_performance_acceptance': True})
    print('Actual full19 firstfail1/raw sealed; independent terminal audit0/own host+container drain0/physical6 free0; zero timings', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('host', 'seal'), required=True)
    args = parser.parse_args()
    host() if args.mode == 'host' else seal()

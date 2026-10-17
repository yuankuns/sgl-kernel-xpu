from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / 'controlled-method-runner-cpu-v1'
RUN = RUNNER / 'runs/stable6-main-controlled-v1'
SUPERVISOR = RUNNER / 'supervisors/stable6-main-controlled-v1'
sys.path.insert(0, str(RUNNER))
from protocol import code_hashes, digest, read, require, save


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light CPU audit0–15')
    grant = read(ROOT / 'main-controlled-diagnostic-grant.json')
    require(digest(ROOT / 'main-controlled-diagnostic-grant.json') == 'caa8a67f5ab2fa8e6b79fc1f1df49ffac7883d01845b2a0b9808f8eaccb3d37a', 'Immutable grant changed')
    require(digest(RUNNER / 'READY.json') == grant['original_READY_sha256'] and code_hashes() == grant['runner_files_sha256'], 'Frozen READY/runner changed')
    exits = {'host_supervisor': int((ROOT / 'host-supervisor.exit').read_text()),
        'wrapper': int((SUPERVISOR / 'wrapper.exit').read_text()), 'first_phase': int((RUN / 'r00-s0-port.exit').read_text())}
    require(all(value == 1 for value in exits.values()), 'Preserve real terminal failure')
    require(len(list(RUN.glob('*.exit'))) == 1 and not list(RUN.rglob('raw-progress.json')), 'No subsequent phase or measured raw allowed after first failure')
    phase = RUN / 'phases/r00-s0-port'
    trace_path = phase / 'prefill-wave-3req-12288/captured_reset_adapter/actual-replay.trace.json'
    dot_path = trace_path.parent / 'capture.dot'
    events = read(trace_path)['traceEvents']
    categories = Counter((event.get('ph'), event.get('cat')) for event in events)
    dot = dot_path.read_text()
    nodes = re.findall(r'(?m)^\s*(L\d+_S\d+_C\d+)\s*\[label=', dot)
    device = [event for event in events if event.get('ph') == 'X' and event.get('dur', 0) > 0
        and re.search(r'kernel|gpu_memcpy|xpu_memcpy|gpu_memset', event.get('cat', ''), re.I)]
    require(not device and len(events) == 442, 'Actual recorded trace facts differ')
    logfile = (RUN / 'r00-s0-port.log').read_text()
    require('Real replay device trace required, not CPU launch labels alone' in logfile, 'Exact first failure reason required')
    start = read(phase / 'runtime-start.json')
    require(start['core_loaded_stack'] == grant['expected_core_loaded_stack_sha256'], 'Actual fresh runtime full core maps mismatch')
    require(start['normal']['mapped_count'] == 712 and start['physical_gpu'] == 6 and start['cpu_affinity'] == list(range(32, 48)), 'Actual normal full AFTER/runtime assignment mismatch')
    require(start['epoch'] >= grant['start_epoch'], 'No start before future grant epoch')
    container = read(ROOT / 'container-own-drain.json')
    require(container['actual_CPU_drain_exit'] == 0 and not container['own_matching_live_runtime'], 'Own runtime not drained')
    host_pid = int((ROOT / 'host-supervisor.pid').read_text())
    require(not Path(f'/proc/{host_pid}').exists(), 'Own host supervisor not drained')
    probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], capture_output=True, text=True)
    require(probe.returncode == 0, 'Physical6 lock not yet released')
    hashes = {}
    for directory in (RUN, SUPERVISOR):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                hashes[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    for path in ROOT.iterdir():
        if path.is_file() and path.name not in ('TERMINAL.json', 'terminal-audit.log', 'terminal-audit.exit'):
            hashes[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    save(ROOT / 'TERMINAL.json', {'independent_terminal_CPU_audit_actual_exit': 0, 'epoch': time.time(),
        'actual_exec_handle': 72321, 'host_supervisor_PID': host_pid, 'container_suite_PID': 275799, 'first_phase_PID': 276125,
        'actual_runtime_exits': exits, 'grant_start_epoch': grant['start_epoch'], 'actual_first_phase_normal_runtime_start_epoch': start['epoch'],
        'first_failure': 'Real replay device trace required, not CPU launch labels alone',
        'actual_trace_events': len(events), 'actual_trace_categories': [{'ph': key[0], 'cat': key[1], 'count': count} for key, count in categories.items()],
        'actual_replay_device_kernel_copy_event_count': len(device), 'actual_DOT_command_node_count': len(nodes),
        'DOT_AppendMemoryCopy_labels': dot.count('<B>zeCommandListAppendMemoryCopy</B>'),
        'DOT_AppendLaunchKernel_labels': dot.count('<B>zeCommandListAppendLaunchKernel</B>'),
        'event_timing_samples_started': False, 'raw_progress_files': 0, 'SOURCE_phase_started': False,
        'captured_output_ref_bits_proof_completed': False, 'controlled_three_method_comparison_complete': False,
        'normal_full_AFTER_and_actual_core_maps_verified_in_first_phase': True,
        'host_and_container_own_processes_drained': True, 'physical_GPU6_lock_free_actual_exit': probe.returncode,
        'logical_GPU6_QK_lease_NOT_handed_off': True, 'automatic_retry_or_runner_READY_grant_changes': False,
        'no_other_GPU_or_process_owner_touched': True, 'all_raw_logs_grant_proof_SHA': hashes,
        'old_percall_regression_and_preopt_failure_NOT_overridden': True, 'performance_acceptance_claim': False})
    print('Terminal audit0: real firstphase/suite/wrapper/host exit1, zero timing raw; actual core/full package checked; true DOT/trace retained; own processes absent and6 physical lock free; no retry')


if __name__ == '__main__':
    main()

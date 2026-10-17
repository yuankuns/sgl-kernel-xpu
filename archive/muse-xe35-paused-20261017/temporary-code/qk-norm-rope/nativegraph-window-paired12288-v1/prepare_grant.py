import os
from pathlib import Path
import subprocess
import time

from contract import ROOT, ROLES, VERSION, case, digest, hashes, read, require, save, unchanged, validate
from runner import live_builders


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light publication CPU0–15')
    require(not (ROOT / 'main-paired-trace-grant.json').exists(), 'New immutable grant once only')
    unchanged()
    ready = read(ROOT / 'CPU_READY.json')
    require(ready['actual_CPU_prepare_exit'] == 0 and ready['driver_files_sha256'] == hashes(), 'Frozen new CPU READY required')
    old = ROOT.parent / 'controlled-gpu6-release-v1'
    terminal = read(old / 'TERMINAL.json')
    require(terminal['independent_terminal_CPU_audit_actual_exit'] == 0 and terminal['host_and_container_own_processes_drained'], 'Old own runtime drain0 required')
    old_host = terminal['host_supervisor_PID']
    require(not Path(f'/proc/{old_host}').exists(), 'Old own host still exists; stop for review')
    current = live_builders()
    require(not current, 'Current compiler/package-manager scan must be empty')
    probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], capture_output=True, text=True)
    require(probe.returncode == 0, 'GPU6 physical lock must be free')
    epoch = time.time()
    drain = {'actual_exit': 0, 'epoch': epoch, 'compiler_and_package_manager_processes': current,
        'old_owned_host_PID_absent': old_host, 'old_owned_container_drain_sha256': digest(old / 'container-own-drain.json'),
        'new_CPU_old_container_absence_proof_sha256': digest(ROOT / 'CPU-old-own-drain.json'),
        'physical_GPU6_lock_free_exit': probe.returncode}
    save(ROOT / 'new-prelaunch-drain.json', drain)
    plan, fields, fixture = case()
    value = {'version': VERSION, 'main_authorized': True, 'issuer': 'main explicit user authorization delegated publication',
        'accuracy_trace_diagnostic_only': True, 'performance_authorized': False, 'gpu': 6, 'runtime_cpus': list(range(32, 48)),
        'roles': list(ROLES), 'calls_per_graph': 32, 'case': fixture, 'providers': plan['providers'],
        'driver_files_sha256': hashes(), 'CPU_READY_sha256': digest(ROOT / 'CPU_READY.json'),
        'core_loaded_stack_sha256': ready['actual_core_files_sha256'],
        'profiler_libraries_sha256': ready['CPU_profiler_libraries_sha256'],
        'new_CPU_provider_source_sha256': digest(ROOT / 'CPU-provider-source.json'),
        'new_CPU_provider_port_sha256': digest(ROOT / 'CPU-provider-port.json'),
        'drain_path': str(ROOT / 'new-prelaunch-drain.json'), 'drain_sha256': digest(ROOT / 'new-prelaunch-drain.json'),
        'old_owned_drain_sha256': digest(old / 'TERMINAL.json'), 'publication_epoch': epoch, 'start_epoch': epoch + 120,
        'strict_original_device_kernel_trace_gate_preserved': True, 'first_failure_stop_no_retry': True,
        'eight_card_claim': False, 'final_full19_performance_scope_NOT_reduced': True}
    validate(value, True)
    save(ROOT / 'main-paired-trace-grant.json', value)
    print(f'New main once grant actual0 start={value["start_epoch"]} SHA={digest(ROOT / "main-paired-trace-grant.json")}', flush=True)


if __name__ == '__main__':
    main()

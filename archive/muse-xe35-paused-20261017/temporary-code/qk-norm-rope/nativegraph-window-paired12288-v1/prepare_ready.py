import os
from pathlib import Path
import sys

from contract import ROOT, case, digest, hashes, read, require, save, unchanged


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)) and 'torch' not in sys.modules, 'Light CPU publication0–15')
    require(not (ROOT / 'CPU_READY.json').exists(), 'Immutable CPU READY once only')
    old = unchanged()
    for name in ('CPU-contracts-final', 'CPU-launch-syntax', 'CPU-provider-source', 'CPU-provider-port', 'CPU-provider-wrapper'):
        require((ROOT / f'{name}.exit').read_text().strip() == '0', 'Actual new CPU proof exit0 required')
    require('Ran 24 tests' in (ROOT / 'CPU-contracts-final.log').read_text(), 'All contracts required')
    source, port = read(ROOT / 'CPU-provider-source.json'), read(ROOT / 'CPU-provider-port.json')
    require(source['actual_exit'] == port['actual_exit'] == 0 and not source['XPU_initialized'] and not port['XPU_initialized'], 'Normal CPU provider proof required')
    require(source['frozen_code_sha256'] == port['frozen_code_sha256'] == hashes(), 'Code modified during/after live CPU proof')
    require(source['current_initialized_core_files_sha256_verified'] == port['current_initialized_core_files_sha256_verified'], 'Actual whole shared core SHA differs')
    require(source['actual_CPU_core_maps_sha256'] == port['actual_CPU_core_maps_sha256'], 'Actual CPU mapped core differs')
    require(source['actual_CPU_profiler_libraries_sha256'] == port['actual_CPU_profiler_libraries_sha256'], 'Actual CPU profiler mapped libraries differ')
    require(source['normal']['mapped_count'] == 155 and port['normal']['mapped_count'] == 712, 'Complete original SOURCE/genuine full AFTER mapped inventory')
    container = read(ROOT / 'CPU-old-own-drain.json')
    require(container['actual_exit'] == 0 and not container['matching_old_owned_runtime'], 'Old owned container child drain must be0')
    plan, fields, fixture = case()
    save(ROOT / 'CPU_READY.json', {'actual_CPU_prepare_exit': 0, 'CPU_contract_tests': 24, 'driver_files_sha256': hashes(),
        'CPU_contract_test_code_sha256': digest(ROOT / 'test_contracts.py'), 'scope': plan['scope'], 'fixture': fixture,
        'providers': plan['providers'], 'actual_core_files_sha256': source['current_initialized_core_files_sha256_verified'],
        'CPU_profiler_libraries_sha256': source['actual_CPU_profiler_libraries_sha256'],
        'normal_SOURCE155_AFTER712_maps_current_verified': True, 'current_original_inputs_and_golden_SHA_verified': True,
        'old_READY_code_grant_failure_unchanged': True, 'old_three_methods19case_final_scope_NOT_reduced': True,
        'new_graph_capture_sentinel_device_trace_accuracy_proof': 'PENDING_NEW_ONCE_MAIN_GRANT',
        'no_performance_samples_or_events': True, 'no_kernel_math_tolerance_or_runtime_replacement': True})
    print(f'CPU_READY actual0/24 contracts/normal155+712/fullinputs/coreSHA; SHA={digest(ROOT / "CPU_READY.json")}', flush=True)


if __name__ == '__main__':
    main()

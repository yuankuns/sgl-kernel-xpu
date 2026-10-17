import ast
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from protocol import SLICE, VERSION, code_hashes, digest, load_plan, read, require, save


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'CPU-only allocation0–15')
    require('torch' not in sys.modules, 'CPU audit does not import Torch')
    require(not (SLICE / 'runs').exists() and not (SLICE / 'supervisors').exists(), 'No future GPU suite was launched')
    for path in SLICE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for name in ('cpu-contracts-final', 'cpu-ready-check-final', 'bash-syntax-final'):
        require((SLICE / f'{name}.exit').read_text().strip() == '0', f'Actual CPU {name} failure')
    require((SLICE / 'launcher-no-grant.exit').read_text().strip() == '2', 'No-grant launcher must fail closed')
    require('Ran 25 tests' in (SLICE / 'cpu-contracts-final.log').read_text(), 'Complete contract suite required')
    plan, fields, cases = load_plan()
    checks = read(SLICE / 'cpu-ready-check-final.json')
    require(checks['actual_runner_files_SHA'] == code_hashes() and checks['GPU_proof_NOT_executed'], 'Actual final CPU runner hashes mismatch')
    for name, checksum in plan['framework_and_source_file_SHA'].items():
        from runner import local

        require(digest(local(name)) == checksum, f'Frozen framework/source file changed: {name}')
    ready = {'status': 'CPU_CONTRACT_READY_ONLY', 'version': VERSION, 'epoch': time.time(), 'CPU_affinity': list(range(16)),
        'actual_CPU_contract_exit': 0, 'CPU_contract_tests': 25, 'actual_CPU_ready_check_exit': 0, 'bash_syntax_exit': 0,
        'launcher_no_grant_actual_exit': 2, 'runner_files_sha256': code_hashes(), 'test_contracts_sha256': digest(SLICE / 'test_contracts.py'),
        'reviewed_plan_fields': fields, 'providers': plan['providers'], 'scope': plan['scope'],
        'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in cases},
        'cases': 19, 'BAAB_rounds': 3, 'phases': 12, 'samples_per_method': 21, 'calls_per_sample': 32,
        'external_GPU_grant_created': False, 'GPU_or_compile_authorized': False, 'Torch_imported': False,
        'current_GPU6_and_CPU32_47_owner': 'Mill W4', 'stable_performance_hold_not_released_here': True,
        'GPU_capture_nodes_actual_kernel_trace_full_reference_and_bits': 'PENDING_NEW_MAIN_GRANT_AND_ACTUAL_RUN',
        'old_regression_confirmed_7_3491_percent_NOT_overridden': True, 'preopt_strictfail3508_NOT_repaired': True,
        'new_physical_lock_or_runtime_drain_claim': False,
        'commands': ['PYTHONDONTWRITEBYTECODE=1 taskset -c 0-15 python3 test_contracts.py',
            'PYTHONDONTWRITEBYTECODE=1 taskset -c 0-15 python3 runner.py --mode cpu-check', 'bash -n launch.sh',
            'taskset -c 0-15 bash launch.sh', 'PYTHONDONTWRITEBYTECODE=1 taskset -c 0-15 python3 audit_cpu.py']}
    save(SLICE / 'READY.json', ready)
    files = {str(path.relative_to(SLICE)): {'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in sorted(SLICE.iterdir()) if path.is_file() and path.name not in ('cpu-final-audit.json', 'cpu-final-audit.log', 'cpu-final-audit.exit', 'write-set.json')}
    save(SLICE / 'write-set.json', {'only_directory': str(SLICE), 'paths': sorted(files),
        'additional_audit_output_paths': ['write-set.json', 'cpu-final-audit.json', 'cpu-final-audit.log', 'cpu-final-audit.exit'],
        'no_old_raw_scripts_inputs_refs_PR_or_global_docs_modified': True})
    save(SLICE / 'cpu-final-audit.json', {'actual_audit_exit': 0, 'CPU_affinity': list(range(16)), 'files_sha256': files,
        'write_set_sha256': digest(SLICE / 'write-set.json'), 'runner_ready_sha256': digest(SLICE / 'READY.json'),
        'Torch_GPU_compile_not_executed': True, 'actual_GPU_proof_PENDING': True})
    print('CPU-only final audit actual0: 25 contracts, all19/12 phases, frozen driver SHA; no GPU grant/run/compile')


if __name__ == '__main__':
    main()

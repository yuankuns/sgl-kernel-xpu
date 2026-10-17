import ast
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from protocol import METHODS, REPEATS, SAMPLES, SLICE, code_hashes, digest, load_plan, paired_binding, phase_plan, read, require, save


require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'CPU0–15 only')
require('torch' not in sys.modules, 'Light AST/audit only, no Torch/XPU import')
require(not (SLICE / 'READY.json').exists(), 'New immutable READY cannot be overwritten')
for name in ('cpu-contracts-final.exit', 'cpu-ready-check.exit', 'bash-syntax.exit', 'cpu-backend-probe.exit'):
    require((SLICE / name).read_text().strip() == '0', 'Actual CPU gate failure')
require('Ran 25 tests' in (SLICE / 'cpu-contracts-final.log').read_text(), 'All25 CPU contracts required')
backend = read(SLICE / 'cpu-backend-probe.json')
require(backend['actual_exit'] == 0 and backend['XPU_initialized'] is False and backend['profiler_session_started'] is False, 'Actual installed backend CPU probe')
old = SLICE.parent / 'controlled-method-runner-cpu-v1'
old_ready = read(old / 'READY.json')
require(digest(old / 'READY.json') == '5ed20090ee1e151a1e9198d7eb4ac242fadd07e35152a4a6a2147ee0dbfa3e38', 'Old READY unchanged')
require(all(digest(old / name) == value for name, value in old_ready['runner_files_sha256'].items()), 'Old script changed')
require(read(SLICE.parent / 'controlled-gpu6-release-v1/TERMINAL.json')['actual_runtime_exits'] ==
    {'host_supervisor': 1, 'wrapper': 1, 'first_phase': 1}, 'Old trace1 must remain1')
for path in SLICE.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
plan, fields, cases = load_plan()
require(len(cases) == 19 and len(phase_plan(cases)) == 12 and SAMPLES == 21 and REPEATS == 32, 'Original static scope')
paired = paired_binding()
files = {path.name: digest(path) for path in SLICE.iterdir() if path.is_file()}
save(SLICE / 'READY.json', {'actual_exit': 0, 'CPU_contracts': 25, 'CPU_only': True, 'GPU_authorized': False,
    'runtime_GPU_proof_NOT_executed': True, 'runtime_performance_NOT_executed': True,
    'cases': 19, 'actual_config_semantic_synthetic': 17, 'old_aggregate_representatives': 2,
    'BAAB_rounds': 3, 'timing_phases': 12, 'independent_proof_processes': 12,
    'methods': METHODS, 'samples': SAMPLES, 'repeats': REPEATS,
    'runner_files_sha256': code_hashes(), 'plan_SHA': fields['plan_SHA'], 'fixture_plan_SHA': fields['fixture_plan_SHA'],
    'all_original_fixture_SHA': {case['name']: case['fixture']['sha256'] for case in cases},
    'paired12288_success_binding': paired, 'paired_success_NOT_all19': True,
    'actual_backend_profiler_state_CPU_probe': backend, 'old_READY_and_scripts_unchanged': True,
    'old_trace_exit1_retained': True, 'proof_and_timing_fresh_process_separation': True,
    'unprofiled_timing_graph_HAS_NO_claimed_device_trace': True,
    'new_main_fresh_grant_required': True, 'runtime_GPU': 6, 'runtime_CPUs': list(range(32, 48)),
    'CPU_preparation_CPUs': list(range(16)), 'all_existing_slice_files_SHA': files})
print('CPU25contracts/backend state/AST/old immutable evidence audit0; v2 READY published; no GPU/performance authorization')

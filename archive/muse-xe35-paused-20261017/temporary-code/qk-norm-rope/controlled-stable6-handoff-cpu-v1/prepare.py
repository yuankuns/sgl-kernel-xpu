import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUNNER = ROOT.parent / 'controlled-method-runner-cpu-v1'
sys.path.insert(0, str(RUNNER))
from protocol import code_hashes, digest, load_plan, phase_plan, read, require


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Only assigned light CPU0–15')
    require('torch' not in sys.modules, 'No Torch/GPU initialization in handoff preparation')
    ready_sha = '5ed20090ee1e151a1e9198d7eb4ac242fadd07e35152a4a6a2147ee0dbfa3e38'
    require(digest(RUNNER / 'READY.json') == ready_sha, 'Original controlled READY must remain frozen')
    ready = read(RUNNER / 'READY.json')
    require(code_hashes() == ready['runner_files_sha256'], 'Frozen scripts cannot change')
    for name in ('cpu-contracts-final', 'cpu-ready-check-final', 'bash-syntax-final', 'cpu-final-audit'):
        require((RUNNER / f'{name}.exit').read_text().strip() == '0', 'Original CPU proof must remain actual0')
    plan, fields, cases = load_plan()
    require(len(cases) == 19 and len(phase_plan(cases)) == 12, 'Keep all19/three BAAB')
    value = {'actual_light_CPU_preparation_exit': 0, 'cpu_affinity': list(range(16)), 'original_READY_sha256': ready_sha,
        'frozen_runner_files_sha256': code_hashes(), 'plan_sha256': fields['plan_SHA'], 'fixture_plan_sha256': fields['fixture_plan_SHA'],
        'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in cases}, 'providers': plan['providers'],
        'requested_future_physical_GPU': 6, 'GPU6_handoff_or_grant_received': False, 'new_grant_created': False,
        'Torch_GPU_compile_wrapper_lock_started': False, 'Mill_drain_NOT_repeated': True,
        'runtime_CPUs_NOT_claimed': True, 'CPU32_47_NOT_used': True,
        'state': 'WAIT_MAIN_EXPLICIT_HANDOFF_FRESH_QUIET_AND_GRANT',
        'required_main_release': {'version': 'qk-controlled-method-diagnostic-v1', 'main_authorized': True,
            'stable_gpu_exclusive_owner': 'QK controlled diagnostic', 'runtime_cpu_owner': 'QK controlled diagnostic',
            'gpu': 6, 'runtime_cpus': 'separate explicit main assignment, not inferred from prior32–47',
            'start_epoch': 'fresh future epoch after actual quiet/drain; publication+120 coordinated',
            'drain_evidence': 'new exact path/SHA, actual_audit_exit0, matching_live_compiler_processes{}, matching_live_package_manager_processes{}',
            'expected_core_loaded_stack_sha256': 'actual matching loaded Torch/SYCL/UR/LevelZero full core maps, not container-name inference',
            'bindings': 'exact existing runner4SHA/plan/fixtures/providers plus immutable new release_id; no old W4/QK grant reuse'},
        'after_grant_only': ['external grant schema and B8 preflight before wrapper', 'normal complete SOURCE/AFTER package/mapped natives SHA',
            'real3.5/XPU availability; actual input/output bits/reference; real backend graph nodes and replay kernel trace',
            'cool5/shape2, three separate raw columns, all19/3BAAB/21x32; no automatic pass'],
        'no_new_runtime_or_whole_package_validation_claim_from_this_light_CPU_check': True}
    (ROOT / 'CPU-handoff-preparation.json').write_text(json.dumps(value, indent=2) + '\n')
    print('Light CPU handoff preparation actual0; frozen READY/4scripts/all19 unchanged; wait explicit main stable6 lease/quiet/grant; no GPU lock or Mill drain')


if __name__ == '__main__':
    main()

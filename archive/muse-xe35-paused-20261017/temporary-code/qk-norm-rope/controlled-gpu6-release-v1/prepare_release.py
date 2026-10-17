import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parents[1]
RUNNER = ROOT.parent / 'controlled-method-runner-cpu-v1'
sys.path.insert(0, str(RUNNER))
from protocol import code_hashes, digest, load_plan, read, require, save, validate_release
from runner import live_builders, scope


def main():
    scope()
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Launcher/lightCPU0–15 only')
    require(not (ROOT / 'main-controlled-diagnostic-grant.json').exists(), 'One immutable new grant; no automatic replacement')
    require(digest(RUNNER / 'READY.json') == '5ed20090ee1e151a1e9198d7eb4ac242fadd07e35152a4a6a2147ee0dbfa3e38', 'Original READY changed')
    ready = read(RUNNER / 'READY.json')
    require(code_hashes() == ready['runner_files_sha256'], 'Frozen runner changed')
    drain_path = ARTIFACT / 'PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1.json'
    quiet_path = ARTIFACT / 'MAINTENANCE_QUIET_AFTER_ROUTING_v4.json'
    w4_path = ARTIFACT / 'mxfp4-w4a4/official-performance-v4-gpu6-v2-post/W4_TERMINAL_PRECISION_PERFORMANCE_DRAIN_v2.json'
    for name in ('PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1', 'MAINTENANCE_QUIET_AFTER_ROUTING_v4'):
        require((ARTIFACT / f'{name}.exit').read_text().strip() == '0', 'Main drain/quiet actual exit0 required')
    drain = read(drain_path)
    quiet = read(quiet_path)
    w4 = read(w4_path)
    require(drain['actual_audit_exit'] == 0 and drain['matching_live_compiler_processes'] == {} and drain['matching_live_package_manager_processes'] == {}, 'Main fresh drain must be empty')
    require(quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60 and all(sample['active_processes'] == {} for sample in quiet['samples']), 'Main continuous maintenance quiet required')
    require(digest(quiet_path) == drain['quiet_evidence_sha256'], 'Main quiet/drain binding mismatch')
    require(w4['CPU_terminal_evidence_actual_exit'] == 0 and w4['true_exec_supervisor_exit'] == 5 and w4['GPU6_physical_lock_free'], 'W4 true failure/drain evidence must remain unchanged')
    source = read(ROOT / 'CPU-core-source.json')
    port = read(ROOT / 'CPU-core-port.json')
    for role in ('source', 'port'):
        require((ROOT / f'CPU-core-{role}.exit').read_text().strip() == '0', 'Actual current normal provider CPU proof required')
    require(source['actual_exit'] == port['actual_exit'] == 0 and not source['XPU_initialized'] and not port['XPU_initialized'], 'CPU normal imports must not initialize XPU')
    cpu_stack = source['CPU_actual_core_loaded_stack_sha256']
    require(cpu_stack == port['CPU_actual_core_loaded_stack_sha256'], 'Both normal providers actual core loaded stack differs')
    candidates = source['read_only_existing_main_GPU7_process_maps'] + port['read_only_existing_main_GPU7_process_maps']
    stack = None
    selected = None
    for candidate in candidates:
        actual = candidate['actual_core_loaded_stack_sha256']
        if all(actual.get(path) == checksum for path, checksum in cpu_stack.items()):
            stack = actual
            selected = candidate
            break
    if stack is None and all(any(token in path for path in cpu_stack) for token in ('libtorch_xpu', 'libsycl', 'libur_adapter', 'libze_')):
        stack = cpu_stack
        selected = {'actual_CPU_normal_provider_maps': True}
    require(stack is not None, 'No actual shared initialized driver/core stack proof; do not fabricate expected map')
    for path, checksum in stack.items():
        host_path = Path(path.replace('/root/', '/home/syk/', 1))
        require(host_path.exists() and digest(host_path) == checksum if path.startswith('/root/') else True, 'Mapped project core file changed')
    plan, fields, cases = load_plan()
    current = live_builders()
    require(not current, 'Final prelaunch maintenance scan nonempty; stop before GPU')
    probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], capture_output=True, text=True)
    require(probe.returncode == 0, 'GPU6 physical lock not free; do not launch')
    kernel_stack = {'running_kernel': os.uname().release, 'loaded_modules': {'xe': {
        name: (Path('/sys/module/xe') / name).read_text().strip() for name in ('version', 'srcversion', 'initstate')}}}
    require(kernel_stack == drain['loaded_stack'], 'Actual kernel/xe stack changed after main quiet')
    epoch = time.time()
    release_id = 'stable6-main-controlled-v1'
    value = {'version': 'qk-controlled-method-diagnostic-v1', 'main_authorized': True,
        'issuer': 'main explicit user authorization; QK delegated immutable publication', 'release_id': release_id,
        'publication_epoch': epoch, 'start_epoch': epoch + 120, 'gpu': 6,
        'stable_gpu_exclusive_owner': 'QK controlled diagnostic', 'runtime_cpu_owner': 'QK controlled diagnostic',
        'runtime_cpus': list(range(32, 48)), 'scope': plan['scope'], 'plan_sha256': fields['plan_SHA'],
        'fixture_plan_sha256': fields['fixture_plan_SHA'], 'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in cases},
        'runner_files_sha256': code_hashes(), 'providers': plan['providers'], 'original_READY_sha256': digest(RUNNER / 'READY.json'),
        'drain_evidence_path': str(drain_path), 'drain_evidence_sha256': digest(drain_path),
        'maintenance_quiet_path': str(quiet_path), 'maintenance_quiet_sha256': digest(quiet_path),
        'W4_terminal_drain_path': str(w4_path), 'W4_terminal_drain_sha256': digest(w4_path),
        'expected_core_loaded_stack_sha256': stack, 'actual_core_stack_probe_source_sha256': digest(ROOT / 'CPU-core-source.json'),
        'actual_core_stack_probe_port_sha256': digest(ROOT / 'CPU-core-port.json'), 'read_only_actual_initialized_core_stack_origin': selected,
        'actual_loaded_kernel_stack': kernel_stack, 'compiler_and_package_manager_final_scan': current,
        'GPU6_final_physical_lock_probe_exit': probe.returncode, 'first_failure_stop_no_retry': True,
        'controlled_diagnostic_only_NOT_performance_acceptance': True, 'eight_card_concurrency_claim': False}
    validate_release(value, plan, fields, cases, code_hashes(), epoch, '6', list(range(32, 48)), True)
    save(ROOT / 'final-prelaunch.json', {'actual_exit': 0, 'epoch': epoch, 'matching_live_compiler_processes_and_package_managers': current,
        'GPU6_lock_free_actual_exit': probe.returncode, 'loaded_kernel_stack': kernel_stack, 'CPU_affinity': list(range(16)),
        'new_main_GPU6_handoff_explicitly_authorized': True, 'other_GPU_locks_NOT_probed': True})
    save(ROOT / 'main-controlled-diagnostic-grant.json', value)
    print(json.dumps({'actual_exit': 0, 'release_id': release_id, 'start_epoch': value['start_epoch'],
        'grant_sha256': digest(ROOT / 'main-controlled-diagnostic-grant.json'), 'physical_GPU': 6, 'runtime_CPUs': '32–47'}), flush=True)


if __name__ == '__main__':
    main()

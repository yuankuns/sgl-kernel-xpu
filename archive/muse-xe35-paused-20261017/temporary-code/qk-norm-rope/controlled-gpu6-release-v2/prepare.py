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
from protocol import VERSION, code_hashes, digest, load_plan, paired_binding, read, require, save, validate_release
from runner import live_builders, local, scope, stack

RELEASE_ID = 'stable6-controlled-v2-main-once'
READY_SHA = '7eaa42fa77f1d4969a8f96f9dc9198d83d628de00407dd11652481d327c11d39'
PACKAGE_NAMES = {'pip', 'pip3', 'uv', 'apt', 'apt-get', 'dpkg', 'conda', 'mamba'}
QUIET_TAG = '-prelaunch'


def frozen():
    scope()
    require(digest(V2 / 'READY.json') == READY_SHA, 'Reviewed v2 READY SHA')
    ready = read(V2 / 'READY.json')
    require(code_hashes() == ready['runner_files_sha256'], 'Frozen v2 driver code')
    require(all(digest(V2 / name) == checksum for name, checksum in ready['all_existing_slice_files_SHA'].items()
        if name.endswith(('.py', '.sh'))), 'All frozen v2 code changed')
    paired_binding()
    return ready


def old_container_drain():
    old = read(ROOT.parent / 'controlled-gpu6-release-v1/TERMINAL.json')
    paired = read(ROOT.parent / 'nativegraph-window-paired12288-v1/TERMINAL.json')
    pids = [old['container_suite_PID'], old['first_phase_PID'], *paired['owned_container_PIDs_absent']]
    require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Old owned container PID not absent')
    active = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            cwd = (directory / 'cwd').resolve(strict=True)
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, PermissionError):
            continue
        if str(ROOT.parent) in str(cwd) and any(token in command for token in ('runner.py --mode suite', 'runner.py --mode phase', 'runner.py --mode proof-phase', 'diagnose.py --mode suite', 'diagnose.py --mode phase')):
            active[directory.name] = command
    require(not active, 'Owned prior QK GPU runtime not drained')
    return {'owned_old_container_PIDs_absent': pids, 'matching_owned_GPU_runtime_processes': active}


def quiet(place):
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Quiet scanner CPU0–15')
    frozen()
    began = time.time()
    monotonic = time.monotonic()
    records = []
    while True:
        current = live_builders()
        records.append({'epoch': time.time(), 'compiler_and_package_manager': current})
        if current:
            save(ROOT / f'{place}-maintenance-quiet{QUIET_TAG}.json', {'actual_audit_exit': 1, 'start_epoch': began, 'samples': records})
            require(False, 'Maintenance activity: stop before grant/launch')
        elapsed = time.monotonic() - monotonic
        if elapsed >= 60:
            break
        time.sleep(1)
    drain = old_container_drain() if place == 'container' else old_host_drain()
    save(ROOT / f'{place}-maintenance-quiet{QUIET_TAG}.json', {'actual_audit_exit': 0, 'epoch': time.time(), 'start_epoch': began,
        'continuous_quiet_seconds': elapsed, 'matching_live_compiler_processes': {}, 'matching_live_package_manager_processes': {},
        'samples': records, 'owned_drain': drain})
    print(f'{place} actual continuous quiet{elapsed:.6f}s and own drain0', flush=True)


def old_host_drain():
    old = read(ROOT.parent / 'controlled-gpu6-release-v1/TERMINAL.json')
    paired = read(ROOT.parent / 'nativegraph-window-paired12288-v1/TERMINAL.json')
    pids = [old['host_supervisor_PID'], paired['actual_host_PID']]
    require(all(not Path(f'/proc/{pid}').exists() for pid in pids), 'Old own host still present')
    return {'owned_old_host_PIDs_absent': pids}


def provider(role):
    require(sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Full normal CPU preflight32–47')
    frozen()
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU provider preflight')
    sys.path.insert(0, str(ROOT.parent / 'normal-v4-model-config-accuracy-v2'))
    from run_accuracy_r2 import normal_audit

    normal = normal_audit(role)
    require(not torch.xpu.is_initialized(), 'Normal full import must not initialize XPU')
    expected = read(ROOT.parent / 'nativegraph-window-paired12288-v1/main-paired-trace-grant.json')['core_loaded_stack_sha256']
    for path, checksum in expected.items():
        require(digest(local(path)) == checksum, 'Current initialized core file SHA changed')
    actual = stack()
    require(all(expected.get(path) == checksum for path, checksum in actual.items()), 'CPU mapped core subset changed')
    plan, fields, cases = load_plan()
    for case in cases:
        require(digest(local(case['fixture']['path'])) == case['fixture']['sha256'], 'Original all19 input SHA')
        require(digest(local(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256'], 'All19 original golden SHA')
    save(ROOT / f'CPU-normal-{role}.json', {'actual_exit': 0, 'normal': normal, 'CPU_core_maps_SHA': actual,
        'actual_initialized_core_files_SHA_verified': expected, 'all19_inputs_and_goldens_SHA_verified': True,
        'XPU_initialized': False, 'pid': os.getpid()})
    print(f'Fresh normal {role}/all19 complete inputs+goldens/current core actual0, XPU uninitialized', flush=True)


def grant():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Publisher CPU0–15')
    require(not (ROOT / 'main-fresh-grant.json').exists(), 'One new immutable grant only')
    ready = frozen()
    require((ROOT / 'external-cpu-contracts.exit').read_text().strip() == '0', 'External monitor CPU contracts')
    require((ROOT / 'CPU-normal-wrapper.exit').read_text().strip() == '0', 'Real normal provider wrapper exit')
    require(not live_builders(), 'Final host maintenance scan not empty')
    old_host_drain()
    for place in ('host', 'container'):
        quiet_value = read(ROOT / f'{place}-maintenance-quiet{QUIET_TAG}.json')
        require(quiet_value['actual_audit_exit'] == 0 and quiet_value['continuous_quiet_seconds'] >= 60, 'Fresh actual60s quiet required')
        require(time.time() - quiet_value['epoch'] < 90, 'Quiet evidence too old')
        require((ROOT / f'{place}-maintenance-quiet{QUIET_TAG}.exit').read_text().strip() == '0', 'Real quiet process exit')
    providers = {role: read(ROOT / f'CPU-normal-{role}.json') for role in ('source', 'port')}
    require(all(value['actual_exit'] == 0 and value['XPU_initialized'] is False for value in providers.values()), 'Full CPU normal both0')
    core = providers['source']['actual_initialized_core_files_SHA_verified']
    require(core == providers['port']['actual_initialized_core_files_SHA_verified'], 'Both provider core stack same')
    probe = subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], check=False)
    require(probe.returncode == 0, 'GPU6 lock not free; no grant')
    require(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= 480 * 1024**3, '380GiB authorized raw plus100GiB headroom admission')
    epoch = time.time()
    drain = {'actual_audit_exit': 0, 'epoch': epoch, 'matching_live_compiler_processes': {}, 'matching_live_package_manager_processes': {},
        'host_quiet_sha256': digest(ROOT / f'host-maintenance-quiet{QUIET_TAG}.json'), 'container_quiet_sha256': digest(ROOT / f'container-maintenance-quiet{QUIET_TAG}.json'),
        'physical_GPU6_lock_free_exit': probe.returncode, 'old_host_drain': old_host_drain()}
    save(ROOT / 'fresh-drain.json', drain)
    plan, fields, cases = load_plan()
    value = {'version': VERSION, 'main_authorized': True, 'issuer': 'explicit main user authorization delegated once grant publication',
        'release_id': RELEASE_ID, 'gpu': 6, 'runtime_cpus': list(range(32, 48)), 'stable_gpu_exclusive_owner': 'QK controlled diagnostic',
        'runtime_cpu_owner': 'QK controlled diagnostic', 'scope': plan['scope'], 'plan_sha256': fields['plan_SHA'],
        'fixture_plan_sha256': fields['fixture_plan_SHA'], 'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in cases},
        'runner_files_sha256': code_hashes(), 'providers': plan['providers'], 'expected_core_loaded_stack_sha256': core,
        'paired12288_terminal_sha256': paired_binding()['terminal_sha256'], 'cpu_READY_sha256': READY_SHA,
        'proof_timing_process_separation': True, 'drain_evidence_path': str(ROOT / 'fresh-drain.json'),
        'drain_evidence_sha256': digest(ROOT / 'fresh-drain.json'), 'issued_epoch': epoch, 'start_epoch': epoch + 120,
        'eight_card_concurrency_claim': False, 'performance_automatic_acceptance': False, 'first_failure_stop_no_retry': True,
        'all_frozen_v2_code_sha256': {name: checksum for name, checksum in ready['all_existing_slice_files_SHA'].items()
            if name.endswith(('.py', '.sh'))},
        'new_CPU_normal_sha256': {role: digest(ROOT / f'CPU-normal-{role}.json') for role in providers},
        'external_supervisor_files_sha256': {name: digest(ROOT / name) for name in ('prepare.py', 'supervise.py', 'watch.py')},
        'coordinator_disk_admission': 'QK6 first/380GiB exclusive new raw reservation; Residual4 held',
        'disk_estimate_sha256': digest(ROOT / 'DISK_ADMISSION.json'),
        'resource_policy': {'worker_RSS_bytes': 96 * 1024**3, 'tree_RSS_bytes': 128 * 1024**3,
            'maximum_wall_seconds': 12 * 3600, 'descendant_not_PGID_monitor': True, 'sample_interval_seconds': 2,
            'maximum_new_raw_bytes': 380 * 1024**3, 'minimum_free_disk_bytes': 100 * 1024**3}}
    validate_release(value, plan, fields, cases, code_hashes(), time.time(), '6', list(range(32, 48)), True)
    save(ROOT / 'main-fresh-grant.json', value)
    print(f'Fresh once main grant SHA={digest(ROOT / "main-fresh-grant.json")} issued={epoch} start={epoch+120}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('host-quiet', 'container-quiet', 'provider', 'grant'))
    parser.add_argument('--role', choices=('source', 'port'))
    args = parser.parse_args()
    if args.mode == 'provider':
        require(args.role, 'Both genuine providers explicitly')
        provider(args.role)
    elif args.mode == 'grant':
        grant()
    else:
        quiet(args.mode.split('-')[0])

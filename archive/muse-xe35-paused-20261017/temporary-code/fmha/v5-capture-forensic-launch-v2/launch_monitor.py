import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
FORENSIC = ART / 'v5-capture-forensic-v2'
sys.path.insert(0, str(ART / 'v5-gpu5-source-port-performance-v1'))
from host_scan import snapshot


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_new(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify_inputs():
    freeze = json.loads((FORENSIC / 'CPU_READY_FINAL_SEAL_v2.json').read_text())
    assert freeze['actual_exit'] == 0
    for name, checksum in freeze['file_SHA'].items():
        assert digest(FORENSIC / name) == checksum, name
    assert freeze['roles']['source']['argument_count'] == 29 and freeze['roles']['port']['argument_count'] == 33
    previous = ART / 'v5-capture-forensic-launch-v1'
    terminal = json.loads((previous / 'resource-monitor-terminal.json').read_text())
    assert terminal['actual_suite_exit'] == 2 and terminal['remaining_known_live_descendants'] == {}
    assert json.loads((previous / 'after-suite-drain-lock.json').read_text())['GPU5_actual_flock_free']
    assert json.loads((previous / 'grant-closed.json').read_text())['closed']
    assert all(not Path(f'/proc/{pid}').exists() for pid in terminal['known_start_time_bound_host_PIDs'])
    old_seal = json.loads((previous / 'TERMINAL_SEAL_CPU_v1.json').read_text())
    for name, checksum in old_seal['file_SHA256'].items():
        if name.endswith(('.log', '.exit', '.json')):
            assert digest(ART / name) == checksum, name
    for role in ('source', 'port'):
        assert (ROOT / f'{role}-fresh-CPU-gate.exit').read_text().strip() == '0'
        fresh = json.loads((ROOT / f'{role}-fresh-CPU-gate.json').read_text())
        old = json.loads((FORENSIC / f'{role}-CPU-gate.json').read_text())
        assert fresh['actual_exit'] == 0 and not fresh['GPU_initialized']
        for field in ('mapped_native_SHA', 'input_sha256', 'SOURCE_full_output_sha256', 'canonical_manifest_sha256', 'runner_sha256'):
            assert fresh[field] == old[field], field
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    actual = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35')
    for name, record in candidate['files'].items():
        assert digest(actual / name) == record['sha256'], name
    assert digest(ART / 'v5-gpu5-performance-preparation/performance_driver.py') == 'c96e671fbc392e8d1b4f4c54346cd40a89b82468206b0b8832cef4ea6bf016e7'
    assert digest(ART / 'v5-gpu5-source-port-performance-v1/source-port-r1-B0.json') == 'bab498e81d545e97f063688a11739046477c7aa61e969c2eb975ab8994036384'
    assert all(not Path(f'/proc/{pid}').exists() for pid in (3301680, 3301681, 3302833, 3302838, 3302898, 3303047))
    assert json.loads((ART / 'v5-gpu5-source-port-performance-v1/grant-closed.json').read_text())['closed']
    return candidate


def issue():
    candidate = verify_inputs()
    scan = snapshot()
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    assert free_bytes >= (380 + 100 + 32) * 1024 ** 3
    write_new('admission-before-grant.json', {'actual_exit': 0, 'host_scan': scan, 'df_available_bytes': free_bytes,
        'snapshot_payload_bytes': 27 * 1024 ** 3 + 24 * 796, 'application_disk_bound_bytes': 32 * 1024 ** 3,
        'QK_reserved_bytes': 380 * 1024 ** 3, 'headroom_bytes': 100 * 1024 ** 3,
        'Residual640_reservation_currently_deferred_by_main': True, 'system_core_extra_not_in_app32GiB': True,
        'main_disk_admission': 'explicit QK380 first/Residual640 deferred; FM bounded <100GiB may launch once'})
    plan = json.loads((ART / 'model-config51-v1/plan.json').read_text())
    port = json.loads((ROOT / 'port-fresh-CPU-gate.json').read_text())
    issued = time.time()
    grant = {'phase': 'fmha-v5-source-port-firstcase-two-generations-capture8-forensic-no-timing',
        'gpu': 5, 'runtime_cpus': '16-31', 'issued_epoch': issued, 'epoch': issued + 120,
        'coordinator_release': True, 'closed': False, 'runtime_reuse_permitted': False,
        'NV_handoff_authorized': False, 'performance_acceptance': False, 'no_timing_warmup_events': True,
        'capture_count': 8, 'generations': 2, 'case_name': 'model-prefill-wave1-active3-layer0',
        'roles': ['source', 'port'], 'forensic_retains_all8_outputs': True,
        'full_reference_each_eager_and_all8_replay_outputs': True,
        'workload': {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'},
        'source_digest': candidate['source_digest'], 'canonical_version': 'canonical-v5-icpx',
        'runner_sha256': digest(FORENSIC / 'forensic_runner.py'), 'controller_sha256': digest(FORENSIC / 'pair_controller.py'),
        'host_supervisor_sha256': digest(Path(__file__)),
        'SOURCE_golden_raw_sha256': digest(ART / 'model-config51-v1/source-accuracy.json'),
        'model_plan_sha256': digest(ART / 'model-config51-v1/plan.json'),
        'source_package_manifest_sha256': digest(ART / 'matched-native-icpx-v1/source/normal-package-manifest.json'),
        'canonical_manifest_sha256': port['canonical_manifest_sha256'],
        'compiler_policy_sha256': port['compiler_scope_and_confound']['compiler_policy_sha256'],
        'normal_CPU_gates_sha256': {role: digest(FORENSIC / f'{role}-CPU-gate.json') for role in ('source', 'port')},
        'fresh_CPU_gates_sha256': {role: digest(ROOT / f'{role}-fresh-CPU-gate.json') for role in ('source', 'port')},
        'frozen_code_CPU_ready_sha256': digest(FORENSIC / 'CPU_READY_FINAL_SEAL_v2.json'),
        'registered_schema_CPU_probe_sha256': {role: digest(FORENSIC / f'{role}-registered-schema-CPU.json')
            for role in ('source', 'port')},
        'real_schema_argument_counts': {'source': 29, 'port': 33},
        'v1_to_v2_minimal_diff_sha256': digest(FORENSIC / 'v1-to-v2-runner.patch'),
        'previous_forensic_v1_terminal_seal_sha256': digest(ART / 'v5-capture-forensic-launch-v1/TERMINAL_SEAL_CPU_v1.json'),
        'previous_forensic_v1_terminal_sha256': digest(ART / 'v5-capture-forensic-launch-v1/resource-monitor-terminal.json'),
        'previous_forensic_v1_drain_sha256': digest(ART / 'v5-capture-forensic-launch-v1/after-suite-drain-lock.json'),
        'previous_forensic_v1_grant_closed_sha256': digest(ART / 'v5-capture-forensic-launch-v1/grant-closed.json'),
        'input_bundle_file_sha256': {name: digest(ART / 'model-config51-v1' / name) for name in plan['fixture_files']},
        'full_original_REFERENCE_helper_sha256': digest(ART / 'launch-v3-icpx/source_only_accuracy.py'),
        'original_backend_fixture_helper_sha256': digest(ART / 'launch-v3-icpx/model_semantic_accuracy.py'),
        'old_host_drain_terminal_sha256': digest(ART / 'v5-gpu5-source-port-performance-v1/FAILED_SUITE_TERMINAL_CPU_AUDIT.json'),
        'old_container_drain_sha256': digest(ART / 'v5-gpu5-source-port-performance-v1/OWNED_CONTAINER_DRAIN_CPU_v2.json'),
        'old_failed_raw_sha256': digest(ART / 'v5-gpu5-source-port-performance-v1/source-port-r1-B0.json'),
        'admission_sha256': digest(ROOT / 'admission-before-grant.json'),
        'resource_limits': {'worker_subtree_RSS_bytes': 96 * 1024 ** 3, 'owned_total_tree_RSS_bytes': 128 * 1024 ** 3,
            'wall_seconds': 4 * 60 * 60, 'sampling_seconds': 1, 'application_disk_bytes': 32 * 1024 ** 3},
        'authorization': 'main explicit once-only v2 paired unmeasured causal diagnostic after actual SOURCE29/PORT33 schema CPU0 and final freeze0; not v1 grant reuse; no compiler/kernel/ABI/math changes',
        'source_compiler_scope': 'three tested math native actual fresh icpx; legacy common_ops/non-target GNU history remains'}
    write_new('gpu5-paired-forensic-barrier-v2.json', grant)
    print(json.dumps({'grant': str(ROOT / 'gpu5-paired-forensic-barrier-v2.json'),
        'grant_sha256': digest(ROOT / 'gpu5-paired-forensic-barrier-v2.json'), 'epoch': grant['epoch'], 'GPU_started': False}), flush=True)


def processes():
    records = {}
    page_size = os.sysconf('SC_PAGE_SIZE')
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            status = (directory / 'status').read_text().splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        namespace_ids = next((line.split()[1:] for line in status if line.startswith('NSpid:')), [])
        records[int(directory.name)] = {'ppid': int(fields[1]), 'state': fields[0], 'start_ticks': int(fields[19]),
            'rss_bytes': max(0, int(fields[21])) * page_size, 'command': command, 'NSpid': namespace_ids}
    return records


def descendants(roots, records):
    owned = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if record['ppid'] in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return owned


def disk_bytes(directory):
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file()) if directory.exists() else 0


def terminate_owned(known, records):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        eligible = [pid for pid, start in known.items() if pid in records and records[pid]['start_ticks'] == start
            and records[pid]['state'] not in ('Z', 'X')]
        if eligible:
            subprocess.run(['sudo', '-n', '/bin/kill', f'-{int(sig)}', *map(str, eligible)], check=False)
        time.sleep(5)
        records = processes()


def run():
    grant_path = ROOT / 'gpu5-paired-forensic-barrier-v2.json'
    grant = json.loads(grant_path.read_text())
    assert digest(Path(__file__)) == grant['host_supervisor_sha256']
    write_new('supervisor-start.json', {'supervisor_host_PID': os.getpid(), 'epoch': time.time(),
        'barrier_epoch': grant['epoch'], 'grant_sha256': digest(grant_path), 'GPU_started': False})
    print(json.dumps({'supervisor_host_PID': os.getpid(), 'epoch': grant['epoch'], 'GPU_started': False}), flush=True)
    time.sleep(max(0, grant['epoch'] - time.time()))
    verify_inputs()
    write_new('admission-before-launch.json', snapshot())
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    assert free_bytes >= (100 + 32) * 1024 ** 3
    assert not (FORENSIC / 'future-run-v1').exists()
    container_grant = str(grant_path).replace('/home/syk/', '/root/', 1)
    container_foreensic = str(FORENSIC).replace('/home/syk/', '/root/', 1)
    command = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh', '--gpu-lock', '--gpu', '5',
        '--workdir', str(ART).replace('/home/syk/', '/root/', 1), '--', 'bash', '-c',
        'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
        'export PYTHONDONTWRITEBYTECODE=1 FMHA_CANONICAL_VERSION=canonical-v5-icpx; '
        'exec taskset -c 16-31 /opt/gfx-deps/venv/bin/python "$1/pair_controller.py" '
        '--grant "$2" --gpu-id 5 --output-dir "$1/future-run-v1"', 'bash', container_foreensic, container_grant]
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015', PYTHONDONTWRITEBYTECODE='1')
    known = {}
    peak_worker = peak_tree = peak_disk = 0
    guard = None
    started = time.time()
    with (ROOT / 'suite.log').open('x') as log, (ROOT / 'resource-monitor.jsonl').open('x') as monitor:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_new('wrapper-start.json', {'wrapper_host_PID': child.pid, 'epoch': started, 'command': command})
        print(json.dumps({'wrapper_host_PID': child.pid, 'GPU5_started': True}), flush=True)
        while child.poll() is None:
            records = processes()
            roots = {child.pid}
            role_roots = []
            for pid, record in records.items():
                command_line = record['command']
                if container_grant in command_line and (f'{container_foreensic}/pair_controller.py ' in command_line
                    or f'{container_foreensic}/forensic_runner.py ' in command_line):
                    roots.add(pid)
                    if f'{container_foreensic}/forensic_runner.py ' in command_line:
                        role_roots.append(pid)
            live_retained = {pid for pid, start in known.items() if pid in records and records[pid]['start_ticks'] == start}
            owned = descendants(roots | live_retained, records)
            for pid in owned:
                if pid in records:
                    known[pid] = records[pid]['start_ticks']
            tree_rss = sum(records[pid]['rss_bytes'] for pid in owned if pid in records and records[pid]['state'] not in ('Z', 'X'))
            worker_rss = max((sum(records[pid]['rss_bytes'] for pid in descendants({worker}, records)
                if pid in records and records[pid]['state'] not in ('Z', 'X')) for worker in role_roots), default=0)
            app_disk = disk_bytes(FORENSIC / 'future-run-v1')
            peak_worker, peak_tree, peak_disk = max(peak_worker, worker_rss), max(peak_tree, tree_rss), max(peak_disk, app_disk)
            sample = {'epoch': time.time(), 'worker_subtree_RSS_bytes': worker_rss, 'owned_tree_RSS_bytes': tree_rss,
                'application_disk_bytes': app_disk, 'actual_owned_PIDs': {str(pid): records[pid] for pid in sorted(owned) if pid in records}}
            monitor.write(json.dumps(sample) + '\n')
            monitor.flush()
            limits = grant['resource_limits']
            if worker_rss > limits['worker_subtree_RSS_bytes'] or tree_rss > limits['owned_total_tree_RSS_bytes']:
                guard = 'RSS_budget'
            elif app_disk > limits['application_disk_bytes']:
                guard = 'application_disk_bound'
            elif time.time() - started > limits['wall_seconds']:
                guard = '4h_deadline'
            if guard:
                terminate_owned(known, records)
                break
            time.sleep(1)
        status = child.wait()
    records = processes()
    live = {str(pid): records[pid] for pid, start in known.items() if pid in records
        and records[pid]['start_ticks'] == start and records[pid]['state'] not in ('Z', 'X')}
    if live:
        terminate_owned(known, records)
        records = processes()
        live = {str(pid): records[pid] for pid, start in known.items() if pid in records
            and records[pid]['start_ticks'] == start and records[pid]['state'] not in ('Z', 'X')}
    with (ROOT / 'suite.exit').open('x') as stream:
        stream.write(f'{status}\n')
    terminal = {'epoch': time.time(), 'actual_suite_exit': status, 'guard': guard,
        'peak_worker_subtree_RSS_bytes': peak_worker, 'peak_owned_tree_RSS_bytes': peak_tree,
        'peak_application_disk_bytes': peak_disk, 'elapsed_seconds': time.time() - started,
        'remaining_known_live_descendants': live, 'known_start_time_bound_host_PIDs': known,
        'tree_monitor_method': 'actual container sibling roots by exact frozen runner/controller + grant command, recursive PPID descendants, retained PID/start ticks; not PGID-only',
        'automatic_retry': False, 'NV_handoff_authorized': False}
    write_new('resource-monitor-terminal.json', terminal)
    write_new('after-suite-drain-lock.json', snapshot())
    write_new('grant-closed.json', {'epoch': time.time(), 'closed': True, 'runtime_reuse_permitted': False,
        'issued_grant_sha256': digest(grant_path), 'actual_suite_exit': status, 'guard': guard,
        'GPU5_FM_logical_lease_retained': True, 'NV_handoff_authorized': False})
    print(json.dumps({'actual_suite_exit': status, 'guard': guard, 'owned_drained': not live,
        'GPU5_flock_free': True, 'grant_closed': True, 'automatic_retry': False}), flush=True)
    raise SystemExit(status if status > 0 else 128 - status if status < 0 else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('issue', 'run'))
    args = parser.parse_args()
    issue() if args.action == 'issue' else run()

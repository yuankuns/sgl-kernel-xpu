import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid


DIRECTORY = Path(__file__).resolve().parent
FROZEN = DIRECTORY.parent / 'normal-v5-performance-gpu1-v1'
CONTAINER = Path('/root') / DIRECTORY.relative_to('/home/syk')
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2) + '\n')


def lock_probe():
    with open('/tmp/intel-gpu-1.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return 0


def verify():
    ready = json.loads((DIRECTORY / 'FINAL_CPU_READY.json').read_text())
    assert ready['actual_exit'] == 0 and not ready['GPU_started']
    for key in ('frozen_v1_dependencies_SHA256', 'external_supervisor_SHA256', 'evidence_SHA256'):
        for path, sha in ready[key].items():
            host = Path('/home/syk') / Path(path).relative_to('/root')
            assert digest(host) == sha, host
    free = shutil.disk_usage(DIRECTORY).free
    required = sum((640, 32, 0, 6, 64, 100)) * 2**30
    assert free >= required, {'free_GiB': free / 2**30, 'required_GiB': required / 2**30}
    return ready


def create():
    ready = verify()
    assert not (DIRECTORY / 'host-started.json').exists()
    for path in (FROZEN / 'phases', FROZEN / 'suite.exit', FROZEN / 'PERFORMANCE_ANALYSIS.json'):
        assert not path.exists(), 'Once-only original suite already started: ' + str(path)
    probe = lock_probe()
    save(DIRECTORY / 'HOST_PRE_GRANT.json', {'actual_exit': 0, 'GPU1_physical_lock_probe_exit': probe,
        'old_owned_GPU1_run_present': False, 'old_owned_PID_startticks': [],
        'frozen_v1_never_launched': True, 'epoch': time.time(), 'free_disk_bytes': shutil.disk_usage(DIRECTORY).free,
        'host_idle_shutdown_waiter_not_stopped_or_modified': True})
    cpu = json.loads((FROZEN / 'CPU_READY.json').read_text())
    quiet = json.loads((DIRECTORY / 'QUIET60.json').read_text())
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    issued = time.time()
    assert 0 <= issued - quiet['samples'][-1]['epoch'] <= 600
    stack = {'running_kernel': quiet['loaded_stack']['running_kernel'], 'xe': quiet['loaded_stack']['loaded_modules']['xe']}
    save(DIRECTORY / 'main-grant.json', {'issuer': 'main-coordinator', 'scope': 'nvfp4-normal-v5-performance-gpu1-v1',
        'delegated_authorization': 'latest explicit conditional once-only formal GPU1 three BAAB; numerical disk and CPU gates passed',
        'gpu': 1, 'performance_authorized': True, 'coordinator_release': True, 'canonical_version': 'canonical-v5-icpx',
        'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
        'rounds': 3, 'order': ['source', 'after', 'after', 'source'], 'warmup': 10, 'samples': 20,
        'CPU_affinity': list(range(80, 96)), 'manifest_sha256': cpu['manifest_sha256'], 'source_digest': cpu['source_digest'],
        'CPU_READY_sha256': digest(FROZEN / 'CPU_READY.json'), 'dependencies_SHA256': ready['frozen_v1_dependencies_SHA256'],
        'quiet_evidence': str(CONTAINER / 'QUIET60.json'), 'quiet_evidence_sha256': digest(DIRECTORY / 'QUIET60.json'),
        'running_stack': stack, 'issued_epoch': issued, 'start_epoch': issued + 120,
        'FINAL_CPU_READY_sha256': digest(DIRECTORY / 'FINAL_CPU_READY.json'),
        'current_precision_provider_shape_evidence_SHA256': ready['evidence_SHA256'],
        'external_supervisor_SHA256': ready['external_supervisor_SHA256'],
        'disk_cap_GiB': 64, 'global_free_floor_GiB': 100, 'maximum_wall_hours': 12,
        'process_cap_GiB': 16, 'tree_cap_GiB': 24, 'minimum_available_GiB': 96, 'automatic_retry': False})
    print(json.dumps({'actual_exit': 0, 'grant': str(DIRECTORY / 'main-grant.json'), 'start_epoch': issued + 120}))


def wrapper(stop=False, token=None):
    command = [WRAPPER]
    if not stop:
        command += ['--gpu-lock', '--gpu', '1']
    command += ['--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '5', 'bash', '-c']
    inner = 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
    if stop:
        inner += 'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu; '
        action = '--stop-owned'
    else:
        inner += 'export MUSE_SOURCE_BUILD_TOKEN=' + token + '; '
        action = '--grant ' + str(CONTAINER / 'main-grant.json')
    inner += 'exec /opt/gfx-deps/venv/bin/python -B ' + str(CONTAINER / 'monitor.py') + ' ' + action
    command.append(inner)
    return command


def launch():
    ready = verify()
    grant = json.loads((DIRECTORY / 'main-grant.json').read_text())
    assert grant['FINAL_CPU_READY_sha256'] == digest(DIRECTORY / 'FINAL_CPU_READY.json')
    assert grant['start_epoch'] > time.time()
    fields = Path('/proc/self/stat').read_text().rpartition(')')[2].split()
    save(DIRECTORY / 'host-started.json', {'PID': os.getpid(), 'startticks': int(fields[19]), 'epoch': time.time(),
         'start_epoch': grant['start_epoch'], 'GPU_lock_held': False})
    print(json.dumps({'waiter_PID': os.getpid(), 'startticks': int(fields[19]), 'start_epoch': grant['start_epoch']}), flush=True)
    while time.time() < grant['start_epoch']:
        time.sleep(min(1, grant['start_epoch'] - time.time()))
    verify()
    lock_probe()
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015', PYTHONDONTWRITEBYTECODE='1')
    token = uuid.uuid4().hex
    command = wrapper(token=token)
    save(DIRECTORY / 'wrapper-command.json', command)
    process = None
    try:
        with (DIRECTORY / 'wrapper.log').open('x') as output:
            process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            save(DIRECTORY / 'wrapper-started.json', {'PID': process.pid, 'token': token, 'epoch': time.time()})
            status = process.wait(timeout=12 * 3600 + 180)
    except BaseException:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        with (DIRECTORY / 'host-stop.log').open('x') as output:
            subprocess.run(wrapper(stop=True), env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=90, check=False)
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        status = process.returncode if process else 1
        (DIRECTORY / 'wrapper.exit').write_text(str(status) + '\n')
        raise
    (DIRECTORY / 'wrapper.exit').write_text(str(status) + '\n')
    save(DIRECTORY / 'HOST_RELEASE.json', {'actual_wrapper_exit': status, 'wrapper_PID_absent': not Path(f'/proc/{process.pid}').exists(),
        'GPU1_lock_probe_exit': lock_probe(), 'epoch': time.time(), 'automatic_retry': False})
    (DIRECTORY / 'host-supervisor.exit').write_text(str(status) + '\n')
    raise SystemExit(status)


def interrupted(signum, frame):
    raise RuntimeError('Host interrupted: ' + str(signum))


if __name__ == '__main__':
    os.sched_setaffinity(0, range(80, 96))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('create', 'launch'))
    args = parser.parse_args()
    create() if args.mode == 'create' else launch()

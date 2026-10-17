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
CONTAINER_DIRECTORY = Path('/root') / DIRECTORY.relative_to('/home/syk')
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
CONTAINER_WRAPPER = '/root/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
GIB = 1024 ** 3


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2) + '\n')


def translated(path):
    return Path('/home/syk') / Path(path).relative_to('/root')


def lock_probe():
    with open('/tmp/intel-gpu-11.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return 0


def seal_check():
    ready = json.loads((DIRECTORY / 'CPU_READY.json').read_text())
    assert ready['actual_exit'] == 0 and ready['GPU_initialized'] is False and ready['GRF512_both']
    for path, sha in ready['dependencies_sha256'].items():
        assert digest(translated(path)) == sha, path
    for relative, sha in ready['evidence_sha256'].items():
        assert digest(DIRECTORY / relative) == sha, relative
    assert shutil.disk_usage(DIRECTORY).free > int(2.25 * GIB)
    return ready


def create():
    ready = seal_check()
    assert not (DIRECTORY / 'main-grant.json').exists()
    assert not (DIRECTORY / 'host-started.json').exists()
    probe = lock_probe()
    save(DIRECTORY / 'HOST_PRE_GRANT.json', {'actual_exit': 0, 'GPU11_lock_probe_exit': probe,
         'owned_prior_run_present': False, 'prior_owned_GPU11_PIDs': [], 'new_artifact_once_only': True,
         'normal_provider_source_exit': 0, 'normal_provider_after_exit': 0,
         'epoch': time.time(), 'free_disk_bytes': shutil.disk_usage(DIRECTORY).free,
         'disk_admission_reservation_bytes': int(2.25 * GIB)})
    issued = time.time()
    save(DIRECTORY / 'main-grant.json', {'issuer': 'main-coordinator-delegated', 'scope': DIRECTORY.name,
         'authorization': 'latest explicit conditional once-only representative SOURCE plus normalv5 AFTER GPU11 accuracy grant',
         'gpu': 11, 'accuracy_only': True, 'performance_authorized': False,
         'canonical_version': 'canonical-v5-icpx', 'serving_batch': 8,
         'ISL': 4096, 'OSL': 16, 'TP': 1, 'EP': 1, 'CPU_affinity': list(range(80, 96)),
         'process_cap_GiB': 16, 'tree_cap_GiB': 24, 'minimum_available_GiB': 96,
         'disk_cap_bytes': int(1.25 * GIB), 'disk_headroom_bytes': GIB,
         'issued_epoch': issued, 'start_epoch': issued + 120,
         'ready_sha256': digest(DIRECTORY / 'CPU_READY.json'),
         'dependencies_sha256': ready['dependencies_sha256'],
         'evidence_sha256': ready['evidence_sha256'], 'once_only': True})
    print(json.dumps({'grant': str(DIRECTORY / 'main-grant.json'), 'epoch': issued + 120, 'actual_exit': 0}))


def wrapper(stop=False, token=None):
    command = [WRAPPER]
    if not stop:
        command += ['--gpu-lock', '--gpu', '11']
    command += ['--workdir', '/root', '--', 'bash', '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh', '5', 'bash', '-c']
    action = '--stop-owned' if stop else '--suite ' + str(CONTAINER_DIRECTORY / 'main-grant.json')
    inner = 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
    if stop:
        inner += 'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu; '
    else:
        inner += 'export MUSE_SOURCE_BUILD_TOKEN=' + token + '; '
    inner += 'exec /opt/gfx-deps/venv/bin/python -B ' + str(CONTAINER_DIRECTORY / 'launch.py') + ' ' + action
    command.append(inner)
    return command


def launch():
    ready = seal_check()
    grant = json.loads((DIRECTORY / 'main-grant.json').read_text())
    assert grant['ready_sha256'] == digest(DIRECTORY / 'CPU_READY.json')
    assert grant['dependencies_sha256'] == ready['dependencies_sha256']
    assert grant['start_epoch'] > time.time(), 'No stale epoch launch'
    fields = Path('/proc/self/stat').read_text().rpartition(')')[2].split()
    save(DIRECTORY / 'host-started.json', {'PID': os.getpid(), 'startticks': int(fields[19]),
         'epoch': time.time(), 'start_epoch': grant['start_epoch'], 'GPU_lock_held': False})
    print(json.dumps({'PID': os.getpid(), 'startticks': int(fields[19]), 'epoch': grant['start_epoch'], 'waiter_ready': True}), flush=True)
    while time.time() < grant['start_epoch']:
        time.sleep(min(1, grant['start_epoch'] - time.time()))
    seal_check()
    lock_probe()
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015', PYTHONDONTWRITEBYTECODE='1')
    token = uuid.uuid4().hex
    command = wrapper(token=token)
    save(DIRECTORY / 'wrapper-command.json', command)
    process = None
    try:
        with (DIRECTORY / 'wrapper.log').open('x') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(DIRECTORY / 'wrapper-started.json', {'PID': process.pid, 'epoch': time.time(), 'token': token})
            status = process.wait()
    except BaseException:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        with (DIRECTORY / 'host-interrupt-stop.log').open('x') as output:
            subprocess.run(wrapper(stop=True), env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=90, check=False)
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        status = process.returncode if process else 1
        (DIRECTORY / 'wrapper.exit').write_text(str(status) + '\n')
        raise
    (DIRECTORY / 'wrapper.exit').write_text(str(status) + '\n')
    probe = lock_probe()
    save(DIRECTORY / 'HOST_RELEASE.json', {'actual_wrapper_exit': status, 'wrapper_PID_absent': not Path(f'/proc/{process.pid}').exists(),
         'GPU11_lock_probe_exit': probe, 'epoch': time.time(), 'no_retry': True, 'performance_started': False})
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

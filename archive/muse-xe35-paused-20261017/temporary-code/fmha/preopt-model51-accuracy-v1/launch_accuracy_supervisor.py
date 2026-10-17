import json
import os
import subprocess
import time
from pathlib import Path
from contracts import ROOT, digest, write_json, code_hashes

WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
CONTAINER_ROOT = '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/preopt-model51-accuracy-v1'
SETUP = '''export PYTHONDONTWRITEBYTECODE=1 FMHA_CANONICAL_VERSION=canonical-v5-icpx
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/preopt/site:/root/sglang/python:${PYTHONPATH:-}"
'''


def old_owner_snapshot():
    rows = []
    markers = ('v5-model51-gpu10-preparation/', 'moe-permutation/source-native-icpx-v1-accuracy/', 'moe-permutation/v4-after-accuracy-prewave-v1/')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] not in ('Z', 'X') and any(any(marker in arg for marker in markers) for arg in args) and Path(args[0]).name in ('python', 'python3', 'bash', 'docker'):
                rows.append({'pid': int(entry.name), 'state': stat[0], 'starttick': stat[19], 'arguments': args})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def main():
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    base = [WRAPPER, '--workdir', CONTAINER_ROOT]
    with (ROOT / 'CURRENT_PREFLIGHT.log').open('x') as log:
        preflight = subprocess.run(base + ['--', 'bash', '-c', SETUP + 'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu\nexec taskset -c 96-111 python -B current_provider_preflight.py'], env=environment, stdout=log, stderr=subprocess.STDOUT)
    (ROOT / 'CURRENT_PREFLIGHT.exit').write_text(str(preflight.returncode) + '\n')
    assert preflight.returncode == 0
    owners = old_owner_snapshot()
    lock = subprocess.run(['flock', '-n', '/tmp/intel-gpu-10.lock', 'true']).returncode
    write_json(ROOT / 'GPU10_PREGRANT_DRAIN_LOCK.json', {'epoch': time.time(), 'old_owned_active_host_and_container_PIDs': owners, 'lock10_probe_actual_exit': lock})
    assert not owners and lock == 0
    grant = json.loads((ROOT / 'GRANT_FIELDS_TEMPLATE_NOT_AUTHORIZED.json').read_text())
    assert grant['driver_files_sha256'] == code_hashes()
    grant.update({'coordinator_release': True, 'issued_epoch': time.time(), 'authorization': 'Main explicit one GPU10 accuracy-only authorization; not performance',
                  'current_provider_preflight_sha256': digest(ROOT / 'CURRENT_NORMAL_STAGE_INPUT_MAPS.json'),
                  'drain_lock_proof_sha256': digest(ROOT / 'GPU10_PREGRANT_DRAIN_LOCK.json')})
    grant['epoch'] = grant['issued_epoch'] + 120
    grant_path = ROOT / 'MAIN_FRESH_GPU10_ACCURACY_GRANT.json'
    assert not grant_path.exists()
    write_json(grant_path, grant)
    write_json(ROOT / 'LAUNCH_METADATA.json', {'host_supervisor_pid': os.getpid(), 'grant_epoch': grant['epoch'], 'grant_sha256': digest(grant_path), 'GPU': 10, 'CPU': '96-111'})
    print(json.dumps({'supervisor_pid': os.getpid(), 'future_epoch': grant['epoch'], 'grant_sha256': digest(grant_path)}), flush=True)
    time.sleep(max(0, grant['epoch'] - time.time()))
    command = base + ['--gpu-lock', '--gpu', '10', '--', 'bash', '-c', SETUP + 'export ONEAPI_DEVICE_SELECTOR=level_zero:gpu FMHA_PREOPT_MODEL51_MAIN_GRANTED=1\nexec taskset -c 96-111 python -B resource_runner.py "$1"', '_', CONTAINER_ROOT + '/' + grant_path.name]
    with (ROOT / 'ACCURACY_RAW.log').open('x') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_json(ROOT / 'WRAPPER_LAUNCH.json', {'PID': child.pid, 'command': command, 'epoch': time.time()})
        actual = child.wait()
    (ROOT / 'ACCURACY_SUPERVISOR.exit').write_text(str(actual) + '\n')
    lock = subprocess.run(['flock', '-n', '/tmp/intel-gpu-10.lock', 'true']).returncode
    write_json(ROOT / 'GPU10_TERMINAL_DRAIN.json', {'epoch': time.time(), 'actual_exit': actual, 'wrapper_PID': child.pid,
               'wrapper_absent': not Path('/proc/' + str(child.pid)).exists(), 'lock_probe_exit': lock, 'old_owned_processes': old_owner_snapshot()})
    raise SystemExit(actual)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        write_json(ROOT / 'SUPERVISOR_FIRST_FAILURE.json', {'traceback': traceback.format_exc(), 'GPU_auto_retry': False})
        (ROOT / 'ACCURACY_SUPERVISOR.exit').write_text('1\n')
        raise

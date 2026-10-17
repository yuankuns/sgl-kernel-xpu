import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'controlled-method-runner-cpu-v2'
sys.path.insert(0, str(V2))
from protocol import code_hashes, digest, load_plan, read, require, save, validate_release
from runner import live_builders, scope


def container_path(path):
    return str(path).replace('/home/syk/', '/root/', 1)


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'External launcher CPU0–15')
    scope()
    require(not (ROOT / 'actual-host-supervisor.json').exists(), 'Once supervisor only')
    grant = read(ROOT / 'main-fresh-grant.json')
    for name, checksum in grant['external_supervisor_files_sha256'].items():
        require(digest(ROOT / name) == checksum, 'External supervisor freeze mismatch')
    plan, fields, cases = load_plan()
    validate_release(grant, plan, fields, cases, code_hashes(), time.time(), '6', list(range(32, 48)), True)
    require(not live_builders(), 'Immediate maintenance scanner nonempty; no launch')
    require(subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true']).returncode == 0, 'GPU6 no longer free')
    save(ROOT / 'actual-host-supervisor.json', {'pid': os.getpid(), 'starttick': Path('/proc/self/stat').read_text().split()[21],
        'epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'grant_sha256': digest(ROOT / 'main-fresh-grant.json')})
    os.environ['SGLANG_MUSE_CONTAINER'] = 'codex-muse-bf16-review-20261015'
    wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
    command = [wrapper, '--workdir', container_path(ROOT), '--', 'bash', '-c',
        'export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 0-15 python3 watch.py']
    save(ROOT / 'resource-monitor.command.json', command)
    with (ROOT / 'resource-monitor.log').open('x') as monitor_log, (ROOT / 'host-launch.log').open('x') as launch_log:
        monitor = subprocess.Popen(command, stdout=monitor_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while not (ROOT / 'resource-monitor-ready.json').exists():
            require(monitor.poll() is None and time.monotonic() < deadline, 'Monitor not ready; no GPU launch')
            time.sleep(0.2)
        launch_command = ['taskset', '-c', '0-15', 'bash', str(V2 / 'launch.sh'), str(ROOT / 'main-fresh-grant.json')]
        save(ROOT / 'host-launch.command.json', launch_command)
        began = time.monotonic()
        launch = subprocess.Popen(launch_command, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
        save(ROOT / 'host-launch.pid.json', {'pid': launch.pid, 'starttick': Path(f'/proc/{launch.pid}/stat').read_text().split()[21]})
        aborted = False
        while launch.poll() is None:
            maintenance = live_builders()
            if monitor.poll() is not None or time.monotonic() - began > grant['resource_policy']['maximum_wall_seconds'] or maintenance:
                save(ROOT / 'host-abort.json', {'epoch': time.time(), 'reason': 'resource monitor terminal, 12h exceeded or new maintenance',
                    'matching_live_compiler_or_package_manager_processes': maintenance})
                aborted = True
                time.sleep(7)
                if launch.poll() is None:
                    os.killpg(launch.pid, signal.SIGTERM)
                break
            time.sleep(1)
        try:
            status = launch.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
            status = launch.wait()
        (ROOT / 'host-launch.exit').write_text(f'{status}\n')
        monitor_status = monitor.wait(timeout=30)
        (ROOT / 'resource-monitor.exit').write_text(f'{monitor_status}\n')
    save(ROOT / 'external-supervisor-terminal.json', {'epoch': time.time(), 'actual_launch_exit': status,
        'actual_monitor_exit': monitor_status, 'aborted': aborted, 'no_retry': True, 'performance_automatic_acceptance': False})
    return status if status else monitor_status


if __name__ == '__main__':
    raise SystemExit(main())

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from prepare_cpu import DIRECTORY, FROZEN, policy

sys.path.insert(0, str(policy.ACCURACY))
from process_tree import ProcessTree, budget_reason


CAP = 64 * 2**30
FREE_FLOOR = 100 * 2**30
MAX_SECONDS = 12 * 3600


def storage():
    size = sum(path.stat().st_size for root in (DIRECTORY, FROZEN) for path in root.rglob('*') if path.is_file())
    free = shutil.disk_usage(DIRECTORY).free
    return {'owned_artifact_bytes': size, 'global_free_bytes': free}


def guard(metrics, files, elapsed):
    reason = budget_reason(metrics)
    if files['owned_artifact_bytes'] > CAP:
        reason = '64GiB artifact cap exceeded'
    if files['global_free_bytes'] < FREE_FLOOR:
        reason = '100GiB global free floor breached'
    if elapsed > MAX_SECONDS:
        reason = '12h bounded suite timeout'
    return reason


def verify_final():
    policy.preflight()
    policy.verify_ready()
    ready = json.loads((DIRECTORY / 'FINAL_CPU_READY.json').read_text())
    assert ready['actual_exit'] == 0 and not ready['GPU_started']
    for key in ('external_supervisor_SHA256', 'evidence_SHA256', 'frozen_v1_dependencies_SHA256'):
        for path, sha in ready[key].items():
            assert policy.digest(path) == sha, path
    return ready


def interrupted(signum, frame):
    raise RuntimeError('External supervisor interrupted: ' + str(signum))


def stop_owned():
    record_path = DIRECTORY / 'container-started.json'
    results = []
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        known = record['identities']
        resources = DIRECTORY / 'resources.jsonl'
        if resources.is_file():
            for line in reversed(resources.read_text().splitlines()):
                try:
                    known += json.loads(line)['metrics']['identities']
                    break
                except json.JSONDecodeError:
                    continue
        tracker = ProcessTree(record['PID'], record['token'], known=known)
        results.append(tracker.drain())
    policy.save(DIRECTORY / 'explicit-stop.json', {'results': results, 'all_owned_drained': all(result['drained'] for result in results)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--stop-owned', action='store_true')
    args = parser.parse_args()
    if args.stop_owned:
        stop_owned()
        return
    if not args.grant:
        parser.error('Fresh main grant required')
    verify_final()
    grant = policy.verify_grant(args.grant)
    assert os.environ['ZE_AFFINITY_MASK'] == '1'
    assert time.time() >= grant['start_epoch']
    assert grant['FINAL_CPU_READY_sha256'] == policy.digest(DIRECTORY / 'FINAL_CPU_READY.json')
    assert not policy.maintenance()
    initial_files = storage()
    assert initial_files['owned_artifact_bytes'] < CAP and initial_files['global_free_bytes'] >= FREE_FLOOR
    token = os.environ['MUSE_SOURCE_BUILD_TOKEN']
    environment = dict(os.environ, NV_V5_PERFORMANCE_GPU1_GRANT=str(args.grant), PYTHONDONTWRITEBYTECODE='1')
    command = ['/opt/gfx-deps/venv/bin/python', '-B', str(FROZEN / 'suite.py'), '--main-authorized', '--grant', str(args.grant)]
    policy.save(DIRECTORY / 'suite-command.json', command)
    started = time.monotonic()
    reason = None
    status = 1
    with (DIRECTORY / 'suite.log').open('x') as log, (DIRECTORY / 'resources.jsonl').open('x') as resources:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        tracker = ProcessTree(child.pid, token)
        policy.save(DIRECTORY / 'container-started.json', {'PID': child.pid, 'token': token,
                    'identities': tracker.sample()['identities'], 'supervisor_PID': os.getpid(),
                    'epoch': time.time(), 'grant_sha256': policy.digest(args.grant)})
        try:
            while True:
                metrics = tracker.sample()
                files = storage()
                active = policy.maintenance()
                reason = guard(metrics, files, time.monotonic() - started)
                if active or policy.stack() != grant['running_stack']:
                    reason = 'Frozen maintenance/loaded stack contamination'
                resources.write(json.dumps({'metrics': metrics, 'storage': files, 'maintenance': active}) + '\n')
                resources.flush()
                if reason:
                    tracker.drain()
                    break
                if child.poll() is not None:
                    if tracker.sample()['live']:
                        reason = 'Root terminal with live descendants'
                        tracker.drain()
                    break
                time.sleep(1)
        except BaseException:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            policy.save(DIRECTORY / 'exception-drain.json', tracker.drain())
            child.wait(timeout=30)
            (DIRECTORY / 'observed-suite.exit').write_text(str(child.returncode) + '\n')
            raise
        status = child.wait(timeout=30)
        final = tracker.sample()
        (DIRECTORY / 'observed-suite.exit').write_text(str(status) + '\n')
        policy.save(DIRECTORY / 'container-drain.json', {'actual_suite_exit': status, 'stop_reason': reason,
                    'identities': final['identities'], 'live': final['live'], 'all_owned_drained': not final['live'],
                    'groups_signalled': sorted(tracker.groups_signalled), 'storage': storage(), 'automatic_retry': False})
        if reason or final['live']:
            raise SystemExit(1)
    raise SystemExit(status)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()

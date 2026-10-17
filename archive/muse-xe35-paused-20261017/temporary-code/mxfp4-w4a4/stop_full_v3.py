import json
import os
import signal
import time

from stop_full_v2 import descendants, processes


def main():
    build = '/root/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35/build'
    state = '/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/build-state-full-v3-j8'
    snapshot = processes()
    supervisors = [pid for pid, record in snapshot.items() if build in record['argv'] and state in record['argv'] and any(argument.endswith('/run-incremental-build.py') for argument in record['argv'])]
    if len(supervisors) != 1:
        raise RuntimeError(f'Expected exactly one owned v3 supervisor: {supervisors}')
    supervisor = supervisors[0]
    children = [pid for pid, record in snapshot.items() if record['parent'] == supervisor]
    groups = {snapshot[pid]['group'] for pid in children}
    if len(groups) != 1 or next(iter(groups)) in (os.getpgrp(), snapshot[supervisor]['group']):
        raise RuntimeError(f'Expected one independent v3 child build group: {groups}')
    owned = descendants(snapshot, supervisor) - {supervisor}
    print(json.dumps({'supervisor': supervisor, 'groups': sorted(groups), 'subtree': {pid: snapshot[pid] for pid in owned}}, indent=2), flush=True)
    for sig, timeout in ((signal.SIGINT, 45), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = processes()
            alive = {pid for pid in owned if pid in current and current[pid]['state'] != 'Z'}
            alive |= {pid for pid, record in current.items() if record['group'] in groups and record['state'] != 'Z'}
            if not alive:
                print('PASS: owned v3 group drained, no live Ninja/compiler descendants.', flush=True)
                return
            time.sleep(0.5)
        print(f'Escalating after {sig.name}; live owned pids={sorted(alive)}', flush=True)
    raise RuntimeError(f'Owned v3 subtree remains live: {alive}')


if __name__ == '__main__':
    main()

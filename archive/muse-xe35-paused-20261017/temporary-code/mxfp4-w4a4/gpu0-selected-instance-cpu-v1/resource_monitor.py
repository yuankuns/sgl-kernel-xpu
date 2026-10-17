import json
import os
from pathlib import Path
import signal
import time

from protocol import DIRECTORY, digest, save


def snapshot(root):
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            status = (path / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
            processes[int(path.name)] = {'PPID': int(stat[1]), 'start_ticks': int(stat[19]), 'state': stat[0], 'RSS_bytes': rss}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {root} if root in processes else set()
    while True:
        added = {pid for pid, row in processes.items() if row['PPID'] in owned} - owned
        if not added:
            break
        owned |= added
    return {pid: processes[pid] for pid in owned if processes[pid]['state'] not in ('Z', 'X')}


def tracked_live(tracked):
    current = {}
    for pid, ticks in tracked.items():
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(stat[19]) == ticks and stat[0] not in ('Z', 'X'):
            current[pid] = ticks
    return current


def main():
    grant_path = DIRECTORY / 'main-gpu0-diagnostic-grant.json'
    grant = json.loads(grant_path.read_text())
    assert grant['issuer'] == 'main' and grant['release'] and grant['gpu'] == 0
    assert grant['resource_monitor_SHA256'] == digest(Path(__file__))
    assert grant['worker_RSS_cap_bytes'] == 96 * 1024**3 and grant['tree_RSS_cap_bytes'] == 128 * 1024**3
    assert grant['walltime_cap_seconds'] == 4 * 3600
    assert sorted(os.sched_getaffinity(0)) == list(range(16))
    save(DIRECTORY / 'resource-monitor-start.json', {'PID': os.getpid(), 'epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'grant_SHA256': digest(grant_path)})
    deadline = grant['start_epoch'] + 4 * 3600
    peak_worker = peak_tree = 0
    root = None
    root_ticks = None
    guard = None
    tracked = {}
    root_path = DIRECTORY / 'container-supervisor.json'
    samples = DIRECTORY / 'resource-samples.jsonl'
    while True:
        if root is None and root_path.exists():
            root = int(json.loads(root_path.read_text())['PID'])
        owned = snapshot(root) if root is not None else {}
        tracked.update({pid: row['start_ticks'] for pid, row in owned.items()})
        if owned and root_ticks is None:
            root_ticks = owned[root]['start_ticks']
        if root_ticks is not None and (root not in owned or owned[root]['start_ticks'] != root_ticks):
            break
        worker = max((row['RSS_bytes'] for row in owned.values()), default=0)
        tree = sum(row['RSS_bytes'] for row in owned.values())
        peak_worker = max(peak_worker, worker)
        peak_tree = max(peak_tree, tree)
        disk = sum(path.stat().st_size for path in DIRECTORY.iterdir() if path.is_file())
        with samples.open('a') as stream:
            stream.write(json.dumps({'epoch': time.time(), 'root_PID': root, 'active_owned': owned, 'worker_max_RSS': worker, 'tree_RSS': tree, 'disk_bytes': disk}) + '\n')
        if worker > grant['worker_RSS_cap_bytes'] or tree > grant['tree_RSS_cap_bytes'] or disk > 6 * 1024**3 or time.time() > deadline:
            guard = {'worker_RSS': worker, 'tree_RSS': tree, 'disk_bytes': disk, 'walltime_exceeded': time.time() > deadline, 'owned': owned}
            for pid, row in sorted(owned.items(), key=lambda item: item[0] == root):
                current = snapshot(root)
                if pid in current and current[pid]['start_ticks'] == row['start_ticks']:
                    os.kill(pid, signal.SIGTERM)
            break
        if root is None and (DIRECTORY / 'host-launch.exit').exists():
            break
        time.sleep(1)
    if guard is not None:
        for wait_index in range(15):
            if not tracked_live(tracked):
                break
            time.sleep(1)
        for pid in tracked_live(tracked):
            os.kill(pid, signal.SIGKILL)
    save(DIRECTORY / 'resource-monitor-result.json', {'peak_worker_RSS_bytes': peak_worker, 'peak_tree_RSS_bytes': peak_tree,
         'guard': guard, 'root_PID': root, 'owned_remaining': tracked_live(tracked),
         'original_inner_900s_per_role_guard_retained_stricter_than_external4h': True, 'no_GPU_or_compiler_work_by_monitor': True})
    raise SystemExit(75 if guard is not None else 1 if tracked_live(tracked) else 0)


if __name__ == '__main__':
    main()

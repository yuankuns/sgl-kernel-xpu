import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'controlled-method-runner-cpu-v2'))
from protocol import digest, read, require, save


def snapshot():
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            status = (path / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
            processes[int(path.name)] = {'ppid': int(stat[1]), 'starttick': int(stat[19]), 'RSS_bytes': rss, 'state': stat[0]}
        except (OSError, ValueError, PermissionError):
            continue
    return processes


def descendants(processes, root_pid, known):
    found = {pid for pid, tick in known.items() if pid in processes and processes[pid]['starttick'] == tick}
    if root_pid in processes:
        if root_pid not in known or known[root_pid] == processes[root_pid]['starttick']:
            found.add(root_pid)
    while True:
        children = {pid for pid, record in processes.items() if record['ppid'] in found}
        if children <= found:
            break
        found |= children
    for pid in found:
        known[pid] = processes[pid]['starttick']
    return {pid: processes[pid] for pid in found}


def signal_owned(known, signum):
    current = snapshot()
    for pid, tick in known.items():
        if pid in current and current[pid]['starttick'] == tick:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light external monitor CPU0–15')
    grant = read(ROOT / 'main-fresh-grant.json')
    require(digest(ROOT / 'watch.py') == grant['external_supervisor_files_sha256']['watch.py'], 'Frozen monitor SHA')
    policy = grant['resource_policy']
    suite = ROOT.parent / 'controlled-method-runner-cpu-v2/runs' / grant['release_id'] / 'supervisor.json'
    known = {}
    began = time.monotonic()
    root_pid = None
    maximum_worker = 0
    maximum_tree = 0
    violation = None
    new_raw_bytes = 0
    free_disk_bytes = 0
    next_disk_scan = 0
    save(ROOT / 'resource-monitor-ready.json', {'pid': os.getpid(), 'starttick': snapshot()[os.getpid()]['starttick'],
        'epoch': time.time(), 'policy': policy, 'no_GPU_import': True})
    with (ROOT / 'resource-samples.jsonl').open('x', buffering=1) as output:
        while True:
            if root_pid is None and suite.exists():
                root_pid = read(suite)['pid']
            processes = snapshot()
            owned = descendants(processes, root_pid, known)
            worker = max((record['RSS_bytes'] for record in owned.values()), default=0)
            total = sum(record['RSS_bytes'] for record in owned.values())
            maximum_worker = max(maximum_worker, worker)
            maximum_tree = max(maximum_tree, total)
            elapsed = time.monotonic() - began
            if elapsed >= next_disk_scan:
                new_raw_bytes = sum(path.stat().st_size for directory in (suite.parent, ROOT)
                    if directory.exists() for path in directory.rglob('*') if path.is_file())
                filesystem = os.statvfs(ROOT)
                free_disk_bytes = filesystem.f_bavail * filesystem.f_frsize
                next_disk_scan = elapsed + 10
            output.write(json.dumps({'epoch': time.time(), 'elapsed_seconds': elapsed, 'suite_PID': root_pid,
                'owned_descendants': owned, 'worker_max_RSS_bytes': worker, 'tree_RSS_bytes': total,
                'new_raw_logical_bytes': new_raw_bytes, 'free_disk_bytes': free_disk_bytes}) + '\n')
            if worker > policy['worker_RSS_bytes'] or total > policy['tree_RSS_bytes'] or elapsed > policy['maximum_wall_seconds'] or new_raw_bytes > policy['maximum_new_raw_bytes'] or free_disk_bytes < policy['minimum_free_disk_bytes']:
                violation = {'epoch': time.time(), 'worker_RSS_bytes': worker, 'tree_RSS_bytes': total,
                    'elapsed_seconds': elapsed, 'new_raw_bytes': new_raw_bytes, 'free_disk_bytes': free_disk_bytes,
                    'policy': policy, 'first_failure_no_retry': True}
                save(ROOT / 'resource-violation.json', violation)
                signal_owned(known, signal.SIGTERM)
                time.sleep(5)
                processes = snapshot()
                descendants(processes, root_pid, known)
                signal_owned(known, signal.SIGKILL)
                break
            if (ROOT / 'host-launch.exit').exists() and not owned:
                break
            if (ROOT / 'host-abort.json').exists():
                signal_owned(known, signal.SIGTERM)
                time.sleep(5)
                descendants(snapshot(), root_pid, known)
                signal_owned(known, signal.SIGKILL)
                violation = {'host_abort': True}
                break
            time.sleep(policy['sample_interval_seconds'])
    remaining = {pid: record for pid, record in snapshot().items() if known.get(pid) == record['starttick']}
    save(ROOT / 'resource-monitor-terminal.json', {'actual_exit': 0 if not violation and not remaining else 124,
        'max_worker_RSS_bytes': maximum_worker, 'max_tree_RSS_bytes': maximum_tree, 'known_PID_startticks': known,
        'matching_live_owned_descendants': remaining, 'violation': violation, 'epoch': time.time()})
    return 0 if not violation and not remaining else 124


if __name__ == '__main__':
    raise SystemExit(main())

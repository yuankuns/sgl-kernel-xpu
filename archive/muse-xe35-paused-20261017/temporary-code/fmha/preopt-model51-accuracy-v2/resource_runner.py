import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from contracts import ROOT, write_json


def snapshot(root_pid):
    processes = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            status = (entry / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
            processes[int(entry.name)] = {'ppid': int(stat[1]), 'state': stat[0], 'rss_bytes': rss, 'starttick': int(stat[19])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    descendants = {root_pid}
    while True:
        expanded = descendants | {pid for pid, record in processes.items() if record['ppid'] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    return {str(pid): processes[pid] for pid in sorted(descendants) if pid in processes}


def main():
    assert os.sched_getaffinity(0) == set(range(96, 112))
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, '-B', str(ROOT / 'accuracy_driver.py'), '--run', '--grant', sys.argv[1]], start_new_session=True)
    peak = 0
    samples = 0
    violation = None
    with (ROOT / 'RESOURCE_RSS_RAW.jsonl').open('x') as log:
        while child.poll() is None:
            processes = snapshot(child.pid)
            rss = sum(record['rss_bytes'] for record in processes.values())
            peak = max(peak, rss)
            samples += 1
            log.write(json.dumps({'epoch': time.time(), 'processes': processes, 'rss_bytes': rss}) + '\n')
            log.flush()
            if rss > 96 * 1024 ** 3 or time.monotonic() - started > 4 * 3600:
                violation = '96GiB_worker_RSS' if rss > 96 * 1024 ** 3 else '4hour_runtime_bound'
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(3)
    actual = child.wait()
    write_json(ROOT / 'RESOURCE_TERMINAL.json', {'child_pid': child.pid, 'actual_child_exit': actual,
               'resource_violation': violation, 'sampled_peak_descendant_RSS_bytes': peak, 'samples': samples,
               'worker_limit_GiB': 96, 'tree_limit_GiB': 128, 'compile_tree_running': False,
               'measurement': '3s sampled actual child and recursive descendants VmRSS; not continuous peak'})
    raise SystemExit(actual if actual >= 0 else 128 - actual)


if __name__ == '__main__':
    main()

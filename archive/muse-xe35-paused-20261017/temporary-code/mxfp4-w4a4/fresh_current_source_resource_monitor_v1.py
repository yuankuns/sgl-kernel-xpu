import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def file_budget(processes):
    maximum = 0
    for root in processes:
        arguments = root['argv']
        if not arguments:
            continue
        executable = Path(arguments[0]).name
        is_compile_script = executable == 'cmake' and '-P' in arguments and any(argument.endswith('.Release.cmake') for argument in arguments)
        if executable not in ['icpx', 'icx'] and not is_compile_script:
            continue
        selected = {root['pid']}
        while True:
            expanded = selected | {item['pid'] for item in processes if item['ppid'] in selected}
            if expanded == selected:
                break
            selected = expanded
        maximum = max(maximum, sum(item['rss_bytes'] for item in processes if item['pid'] in selected))
    return maximum


def monitored(stage, name, command, timeout=1800):
    (stage / (name + '.command')).write_text(shlex.join(command) + '\n')
    tracked = {}
    guard = None
    peak_tree = 0
    peak_process = 0
    peak_file = 0
    started = time.monotonic()
    with (stage / (name + '.log')).open('w') as output, (stage / (name + '.resources.jsonl')).open('w') as samples:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record(stage / (name + '.started.json'), {'pid': child.pid, 'epoch': time.time(), 'command': command})
        while child.poll() is None:
            processes = resources(child.pid, tracked)
            total = sum(item['rss_bytes'] for item in processes)
            maximum = max((item['rss_bytes'] for item in processes), default=0)
            file_rss = file_budget(processes)
            peak_file = max(peak_file, file_rss)
            peak_tree = max(peak_tree, total)
            peak_process = max(peak_process, maximum)
            samples.write(json.dumps({'epoch': time.time(), 'tree_rss_bytes': total, 'max_file_subtree_rss_bytes': file_rss, 'processes': processes}) + '\n')
            samples.flush()
            if file_rss > 8 * 1024**3 or maximum > 8 * 1024**3 or total > 16 * 1024**3 or time.monotonic() - started > timeout:
                guard = '8GiB file/compiler subtree or process / 16GiB tree / stage timeout exceeded'
                terminate(processes, signal.SIGTERM)
                time.sleep(3)
                terminate(resources(child.pid, tracked), signal.SIGKILL)
                break
            time.sleep(0.25)
        status = child.wait()
        if resources(child.pid, tracked):
            guard = guard or 'Residual descendants required drain'
            terminate(resources(child.pid, tracked), signal.SIGTERM)
            time.sleep(3)
            terminate(resources(child.pid, tracked), signal.SIGKILL)
        (stage / (name + '.exit')).write_text(str(status) + '\n')
        record(stage / (name + '.resource-summary.json'), {'actual_exit': status, 'guard': guard,
               'peak_file_subtree_rss_bytes': peak_file, 'peak_tree_rss_bytes': peak_tree, 'peak_process_rss_bytes': peak_process,
               'drained': not resources(child.pid, tracked), 'coverage': 'recursive PPID descendants and start-time-bound retained PIDs; not PGID-only'})
        if status or guard:
            raise RuntimeError(f'{name}: actual_exit={status}, guard={guard}')



def attach(support):
    global resources, terminate, record
    resources = support.resources
    terminate = support.terminate
    record = support.record
    support.monitored = monitored

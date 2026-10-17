import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def processes():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            result[int(directory.name)] = {'ppid': int(fields[1]), 'pgrp': int(fields[2]),
                'starttime': int(fields[19]), 'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                'name': (directory / 'comm').read_text().strip(), 'command': command}
        except (OSError, ValueError):
            continue
    return result


def tracked_tree(current, root, known):
    members = {process_id for process_id, stamp in known.items()
               if process_id in current and current[process_id]['starttime'] == stamp}
    if root in current:
        members.add(root)
    while True:
        discovered = {process_id for process_id, item in current.items() if item['ppid'] in members}
        if discovered <= members:
            break
        members |= discovered
    for process_id in members:
        known[process_id] = current[process_id]['starttime']
    return members


def compiler_memory(current, members):
    compilers = {process_id for process_id in members if any(token in current[process_id]['name']
                for token in ('icpx', 'icx', 'clang', 'cc1', 'ocloc'))}
    totals = {}
    for root in compilers:
        subtree = {root}
        while True:
            children = {process_id for process_id in members if current[process_id]['ppid'] in subtree}
            if children <= subtree:
                break
            subtree |= children
        totals[root] = sum(current[process_id]['rss'] for process_id in subtree)
    return totals


def terminate_tree(root, known):
    stopped = set()
    for _ in range(4):
        current = processes()
        members = tracked_tree(current, root, known)
        for process_id in members:
            try:
                os.kill(process_id, signal.SIGSTOP)
                stopped.add(process_id)
            except ProcessLookupError:
                pass
        time.sleep(0.05)
    for process_id in stopped:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sorted(stopped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.command.json').write_text(json.dumps({'command': command, 'cwd': str(args.cwd),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'outer_jobs': 2, 'MAX_JOBS': os.environ.get('MAX_JOBS'),
        'compiler_subtree_limit_bytes': 8 * 1024 ** 3, 'entire_descendant_tree_limit_bytes': 16 * 1024 ** 3,
        'tracking': 'PPID transitive closure plus retained PID/starttime identities; includes separate PGIDs and reparented known descendants'}, indent=2) + '\n')
    started = time.time()
    known, peak_tree, peak_compiler, killed = {}, 0, 0, []
    with prefix.with_suffix('.log').open('w') as log, prefix.with_suffix('.resources.jsonl').open('w') as resources:
        child = subprocess.Popen(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        prefix.with_suffix('.started.json').write_text(json.dumps({'pid': child.pid, 'supervisor_pid': os.getpid(), 'epoch': started}) + '\n')
        while True:
            current = processes()
            members = tracked_tree(current, child.pid, known)
            tree_rss = sum(current[process_id]['rss'] for process_id in members)
            compiler_rss = compiler_memory(current, members)
            largest = max(compiler_rss.values(), default=0)
            peak_tree, peak_compiler = max(peak_tree, tree_rss), max(peak_compiler, largest)
            resources.write(json.dumps({'epoch': time.time(), 'tree_rss': tree_rss, 'max_compiler_subtree_rss': largest,
                'processes': {str(process_id): current[process_id] for process_id in sorted(members)}}) + '\n')
            resources.flush()
            if tree_rss > 16 * 1024 ** 3 or largest > 8 * 1024 ** 3:
                killed = terminate_tree(child.pid, known)
                break
            if child.poll() is not None and not members:
                break
            time.sleep(0.1)
        result = child.wait()
        while tracked_tree(processes(), child.pid, known):
            time.sleep(0.1)
    status = result if result >= 0 else 128 - result
    if killed:
        status = 137
    prefix.with_suffix('.exit').write_text(str(status) + '\n')
    prefix.with_suffix('.terminal.json').write_text(json.dumps({'exit': status, 'child_returncode': result,
        'wall_seconds': time.time() - started, 'peak_tree_rss': peak_tree, 'peak_compiler_subtree_rss': peak_compiler,
        'resource_terminated_pids': killed, 'remaining_known_live': sorted(tracked_tree(processes(), child.pid, known))}, indent=2) + '\n')
    print(json.dumps({'phase': str(prefix), 'exit': status, 'peak_tree_gib': peak_tree / 1024 ** 3,
                      'peak_compiler_gib': peak_compiler / 1024 ** 3}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

from process_tree import ProcessTree, TOKEN_KEY, budget_reason

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
GATE_OLD = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
GATE_NEW = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR\n   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value):
    before, after = ('/root/', '/home/syk/') if str(ROOT).startswith('/home/syk/') else ('/home/syk/', '/root/')
    return Path(value.replace(before, after, 1) if value.startswith(before) else value)


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')
    path.chmod(0o644)


def process_ticks(process_id):
    return Path(f'/proc/{process_id}/stat').read_text().rsplit(')', 1)[1].split()[19]


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def git(source, *arguments):
    return subprocess.check_output(['git', '-c', f'safe.directory={source}', '-C', str(source), *arguments])






def verify_snapshot(config, snapshot):
    source = local_path(config['source_worktree'])
    entries = git(source, 'ls-tree', '-r', '-z', config['source_revision']).split(b'\0')
    expected_paths = set()
    for entry in entries:
        if not entry:
            continue
        metadata_text, name_bytes = entry.split(b'\t', 1)
        mode, kind, blob = metadata_text.decode().split()
        if kind != 'blob':
            raise RuntimeError('Unexpected gitlink in source snapshot')
        name = name_bytes.decode()
        expected_paths.add(name)
        path = snapshot / name
        content = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
        if name == 'cmake/BuildFlags.cmake':
            original = git(source, 'show', config['source_revision'] + ':' + name).decode()
            if original.count(GATE_OLD) != 1 or content != original.replace(GATE_OLD, GATE_NEW).encode():
                raise RuntimeError('Only reviewed two-line IntelLLVM gate allowed')
        elif hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != blob:
            raise RuntimeError('SOURCE body/helper changed: ' + name)
    actual_paths = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*') if path.is_file() or path.is_symlink()}
    if actual_paths != expected_paths:
        raise RuntimeError('Source snapshot inventory changed')
    return {'original_tree': git(source, 'rev-parse', config['source_revision'] + '^{tree}').decode().strip(), 'files_checked': len(expected_paths), 'only_source_delta': ['cmake/BuildFlags.cmake'], 'all_kernel_bodies_helpers_preserved': True}




def stop_group(process, tracker, stem):
    result = tracker.drain()
    save(stem.with_suffix('.drain.json'), result)
    if process.poll() is None:
        process.wait(timeout=10)
    if not result['drained']:
        raise RuntimeError('Owned descendants remain after multi-PGID drain')


def run_command(command, stem, environment, timeout=3600, memory_guard=False):
    stem.with_suffix('.command').write_text(shlex.join(command) + '\n')
    start = time.time()
    token = uuid.uuid4().hex
    environment = dict(environment, **{TOKEN_KEY: token})
    reason = None
    with stem.with_suffix('.log').open('x') as output, stem.with_suffix('.resources.jsonl').open('x') as resources:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        tracker = ProcessTree(process.pid, token)
        save(stem.with_suffix('.started.json'), {'pid': process.pid, 'pgid': process.pid, 'proc_start_ticks': process_ticks(process.pid), 'epoch': start, 'command': command, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'token': token})
        try:
            while True:
                metrics = tracker.sample()
                resources.write(json.dumps(metrics) + '\n')
                resources.flush()
                save(stem.with_suffix('.owned.json'), {'token': token, 'identities': metrics['identities']})
                reason = budget_reason(metrics) if memory_guard else None
                if time.time() - start > timeout:
                    reason = 'bounded command timeout'
                if reason:
                    stop_group(process, tracker, stem)
                    break
                if process.poll() is not None:
                    metrics = tracker.sample()
                    if metrics['live']:
                        reason = 'root exited while owned descendants remained'
                        stop_group(process, tracker, stem)
                    break
                time.sleep(0.25)
        except BaseException:
            previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_group(process, tracker, stem)
            finally:
                stem.with_suffix('.exit').write_text(str(process.poll()) + '\n')
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
            raise
    status = process.wait()
    final = tracker.sample()
    save(stem.with_suffix('.drain.json'), {'drained': not final['live'], 'live_after': final['live'], 'identities': final['identities'], 'groups_signalled': sorted(tracker.groups_signalled)})
    stem.with_suffix('.exit').write_text(str(status) + '\n')
    save(stem.with_suffix('.terminal.json'), {'actual_exit': status, 'elapsed_seconds': time.time() - start, 'stop_reason': reason, 'all_descendants_drained': not final['live'], 'compiler_rss_limit_gib': 16, 'tree_rss_limit_gib': 24, 'available_floor_gib': 96})
    if status != 0 or reason or final['live']:
        raise RuntimeError(f'Actual command failure {status}: {stem}; {reason}')


def setting(text, name):
    match = re.search(r'^set\(' + re.escape(name) + r' (.*)\)(?: #.*)?$', text, re.MULTILINE)
    if not match:
        raise RuntimeError('Missing actual generated compiler metadata: ' + name)
    return match.group(1)














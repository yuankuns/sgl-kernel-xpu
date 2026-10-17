import json
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parent
target = Path('/root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35')
supervisor = 38269


def processes():
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            result[int(entry.name)] = {'pid': int(entry.name), 'ppid': int(fields[1]), 'group': int(fields[2]), 'state': fields[0], 'command': command, 'cwd': os.readlink(entry / 'cwd')}
        except (OSError, IndexError):
            pass
    return result


snapshot = processes()
assert snapshot[supervisor]['cwd'] == str(target)
assert 'run-incremental-build.py --build-dir build --jobs 2 ' in snapshot[supervisor]['command']
descendants = {supervisor}
while True:
    expanded = descendants | {pid for pid, record in snapshot.items() if record['ppid'] in descendants}
    if expanded == descendants:
        break
    descendants = expanded
record = {'supervisor_container_pid': supervisor, 'supervisor_host_pid': 1540796, 'pipeline_host_pid': 1540730, 'terminal_handle': 33762, 'signal': 'SIGTERM to verified own supervisor; runner gracefully interrupts its child process group', 'before': [snapshot[pid] for pid in sorted(descendants)], 'resources_before': Path('/proc/meminfo').read_text(), 'signal_epoch': time.time()}
(root / 'build-j2-stop.json').write_text(json.dumps(record, indent=2) + '\n')
os.kill(supervisor, signal.SIGTERM)
deadline = time.monotonic() + 120
while True:
    current = processes()
    remaining = [current[pid] for pid in descendants if pid in current and current[pid]['state'] != 'Z']
    active_tools = [value for value in current.values() if value['state'] != 'Z' and any(tool in value['command'].split(' ', 1)[0] for tool in ('ninja', 'icx', 'clang', 'cc1plus', 'ocloc', 'compiler_launcher', '/cmake')) and (str(target) in value['command'] or value['cwd'].startswith(str(target)))]
    if not remaining and not active_tools:
        break
    if time.monotonic() > deadline:
        record['remaining'] = remaining
        record['active_tools'] = active_tools
        (root / 'build-j2-stop.json').write_text(json.dumps(record, indent=2) + '\n')
        raise RuntimeError('Own build subtree still active; do not restart Ninja')
    time.sleep(1)
record['own_descendants_ended'] = True
record['active_target_compiler_ninja_after'] = []
record['completed_epoch'] = time.time()
(root / 'build-j2-stop.json').write_text(json.dumps(record, indent=2) + '\n')
print('PASS: own full-build supervisor and complete child tree ended; no target compiler/Ninja active', flush=True)

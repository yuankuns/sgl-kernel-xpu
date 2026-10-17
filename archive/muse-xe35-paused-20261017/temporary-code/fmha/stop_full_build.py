import argparse
import json
import os
from pathlib import Path
import signal
import time


art = Path(__file__).resolve().parent
root = '/root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35'
parser = argparse.ArgumentParser()
parser.add_argument('--root-pid', type=int, required=True)
parser.add_argument('--ninja-pid', type=int, required=True)
args = parser.parse_args()


def processes():
    records = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            records[int(path.name)] = {'ppid': int(fields[1]), 'state': fields[0], 'start': fields[19],
                                       'cmd': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                                       'cwd': str((path / 'cwd').resolve())}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return records


records = processes()
assert f'{art}/build_xe35.sh' in records[args.root_pid]['cmd']
assert records[args.ninja_pid]['cwd'] == root + '/build-xe35'
owned = {args.root_pid}
while True:
    expanded = owned | {pid for pid, record in records.items() if record['ppid'] in owned}
    if expanded == owned:
        break
    owned = expanded
assert args.ninja_pid in owned
snapshot = {'root_pid': args.root_pid, 'ninja_pid': args.ninja_pid, 'stop_epoch': time.time(),
            'processes': {pid: records[pid] for pid in sorted(owned)},
            'memory': Path('/proc/meminfo').read_text()}
(art / 'full-build-v1-stop-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
os.kill(args.ninja_pid, signal.SIGINT)
for pid in sorted(owned - {args.root_pid, args.ninja_pid}, reverse=True):
    try:
        if processes().get(pid, {}).get('start') == records[pid]['start']:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 120
while True:
    current = processes()
    active = {pid: current[pid] for pid in owned if pid in current and current[pid]['start'] == records[pid]['start']
              and current[pid]['state'] != 'Z'}
    if not active:
        break
    if time.monotonic() >= deadline:
        raise RuntimeError(f'old build still active; do not launch successor: {active}')
    time.sleep(1)
remaining = {pid: record for pid, record in processes().items()
             if record['cwd'].startswith(root + '/build-xe35') and record['state'] != 'Z'}
assert not remaining, remaining
snapshot.update(terminal_epoch=time.time(), all_old_processes_terminal=True, residual_active_build_processes=remaining)
(art / 'full-build-v1-stop-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
print(json.dumps({'old_build_processes': len(owned), 'all_terminal': True, 'remaining': remaining}), flush=True)

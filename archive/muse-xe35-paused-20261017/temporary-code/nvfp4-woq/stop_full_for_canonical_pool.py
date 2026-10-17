import json
import os
import pathlib
import signal
import time

artifact = pathlib.Path(__file__).parent
processes = {}
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        status = (entry / 'status').read_text()
        parent = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        processes[int(entry.name)] = {'parent': parent, 'command': command}
    except (OSError, StopIteration):
        continue
roots = [pid for pid, record in processes.items() if record['command'].startswith('bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/build_target_jobs8_link1_v1.sh ')]
assert len(roots) == 1, roots
root = roots[0]
selected = {root}
while True:
    expanded = selected | {pid for pid, record in processes.items() if record['parent'] in selected}
    if expanded == selected:
        break
    selected = expanded
assert processes[root]['command'].startswith('bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/build_target_jobs8_link1_v1.sh ')
for pid in sorted(selected, reverse=True):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
def active(pid):
    try:
        status = pathlib.Path(f'/proc/{pid}/status').read_text()
        return not next(line for line in status.splitlines() if line.startswith('State:')).split()[1] == 'Z'
    except (OSError, StopIteration):
        return False
deadline = time.monotonic() + 30
while any(active(pid) for pid in selected) and time.monotonic() < deadline:
    time.sleep(0.5)
remaining = [pid for pid in selected if active(pid)]
assert not remaining, remaining
record = {
    'reason': 'User canonical full-pool deduplication: stop own full only, preserve GRF512 v2/deps/flags; GPU9 diagnostic/tails untouched',
    'stopped_session': 89273, 'supervisor_pid': 2308000,
    'terminated': [{'pid': pid, **processes[pid]} for pid in sorted(selected)],
    'active_after': remaining, 'meminfo': pathlib.Path('/proc/meminfo').read_text(),
}
(artifact / 'canonical-pool-full-stop-audit.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'terminated_count': len(selected), 'active_after': remaining}))

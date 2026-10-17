import os
from pathlib import Path
import signal
import time

TOKEN_KEY = 'MUSE_SOURCE_BUILD_TOKEN'
COMPILERS = {'icx', 'icpx', 'clang', 'clang++', 'cc1', 'cc1plus', 'ocloc', 'llvm-spirv', 'spirv-link'}


def snapshot():
    records = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / 'stat').read_text()
            fields = text.rsplit(')', 1)[1].split()
            process_id = int(entry.name)
            records[process_id] = {'pid': process_id, 'name': text.split('(', 1)[1].rsplit(')', 1)[0], 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19]), 'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    return records


def token_matches(process_id, token):
    try:
        return (TOKEN_KEY + '=' + token).encode() in Path(f'/proc/{process_id}/environ').read_bytes().split(b'\0')
    except OSError:
        return False


def budget_reason(metrics):
    if metrics['largest_compiler_rss_gib'] > 8:
        return 'compiler RSS exceeded 8GiB'
    if metrics.get('largest_owned_process_rss_gib', 0) > 8:
        return 'unrecognized owned tool RSS exceeded conservative 8GiB cap'
    if metrics['tree_rss_gib'] > 16:
        return 'owned tree RSS exceeded 16GiB'
    if metrics['available_gib'] < 96:
        return 'available memory below 96GiB'
    return None


class ProcessTree:
    def __init__(self, root_pid, token, known=None):
        self.token = token
        self.known = {} if known is None else {(int(item['pid']), int(item['start_ticks'])): item for item in known}
        current = snapshot()
        if root_pid in current:
            record = current[root_pid]
            self.known[(root_pid, record['start_ticks'])] = record
        self.groups_signalled = set()

    def discover(self):
        current = snapshot()
        selected = {process_id for process_id, record in current.items() if (process_id, record['start_ticks']) in self.known}
        minimum_ticks = min((identity[1] for identity in self.known), default=0)
        for process_id, record in current.items():
            if record['start_ticks'] >= minimum_ticks and process_id not in selected and token_matches(process_id, self.token):
                selected.add(process_id)
        changed = True
        while changed:
            changed = False
            for process_id, record in current.items():
                if process_id not in selected and record['ppid'] in selected:
                    selected.add(process_id)
                    changed = True
        for process_id in selected:
            record = current[process_id]
            self.known[(process_id, record['start_ticks'])] = record
        live = {process_id: current[process_id] for process_id in selected if current[process_id]['state'] not in ('Z', 'X')}
        return current, live

    def sample(self):
        current, live = self.discover()
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        compilers = [record for record in live.values() if record['name'] in COMPILERS or record['name'].startswith(('clang', 'icx', 'icpx', 'llvm', 'spirv', 'ocloc'))]
        return {'epoch': time.time(), 'available_gib': available / 1024 ** 3, 'tree_rss_gib': sum(record['rss_bytes'] for record in live.values()) / 1024 ** 3, 'largest_owned_process_rss_gib': max((record['rss_bytes'] for record in live.values()), default=0) / 1024 ** 3, 'largest_compiler_rss_gib': max((record['rss_bytes'] for record in compilers), default=0) / 1024 ** 3, 'compiler_rss_gib': {str(record['pid']): record['rss_bytes'] / 1024 ** 3 for record in compilers}, 'live': list(live.values()), 'identities': list(self.known.values()), 'pgids': sorted({record['pgid'] for record in live.values()})}

    def signal_owned(self, signum):
        current, live = self.discover()
        for group in {record['pgid'] for record in live.values()}:
            members = [record for record in current.values() if record['pgid'] == group and record['state'] not in ('Z', 'X')]
            if members and all((record['pid'], record['start_ticks']) in self.known for record in members):
                try:
                    os.killpg(group, signum)
                    self.groups_signalled.add(group)
                except ProcessLookupError:
                    pass
        for record in live.values():
            try:
                descriptor = os.pidfd_open(record['pid'])
                try:
                    latest = snapshot().get(record['pid'])
                    if latest and latest['start_ticks'] == record['start_ticks']:
                        signal.pidfd_send_signal(descriptor, signum)
                finally:
                    os.close(descriptor)
            except ProcessLookupError:
                pass

    def drain(self):
        self.signal_owned(signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not self.discover()[1]:
                break
            time.sleep(0.1)
            self.signal_owned(signal.SIGTERM)
        kill_deadline = time.monotonic() + 10
        while self.discover()[1] and time.monotonic() < kill_deadline:
            self.signal_owned(signal.SIGKILL)
            time.sleep(0.1)
        final = self.sample()
        return {'drained': not final['live'], 'live_after': final['live'], 'groups_signalled': sorted(self.groups_signalled), 'identities': list(self.known.values())}

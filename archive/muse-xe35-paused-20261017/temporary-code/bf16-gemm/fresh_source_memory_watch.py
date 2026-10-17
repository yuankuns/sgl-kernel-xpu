import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    root_pid = int(sys.argv[1])
    destination = Path(sys.argv[2])
    peak = 0
    with destination.open('x') as stream:
        while Path('/proc', str(root_pid)).exists():
            table = {}
            for directory in Path('/proc').iterdir():
                if not directory.name.isdigit():
                    continue
                try:
                    fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
                    table[int(directory.name)] = {'ppid': int(fields[1]), 'state': fields[0],
                        'rss_bytes': int((directory / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE'),
                        'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
            selected = {root_pid}
            while True:
                children = {pid for pid, record in table.items() if record['ppid'] in selected}
                expanded = selected | children
                if expanded == selected:
                    break
                selected = expanded
            records = {pid: table[pid] for pid in selected if pid in table and table[pid]['state'] != 'Z'}
            total = sum(record['rss_bytes'] for record in records.values())
            peak = max(peak, total)
            stream.write(json.dumps({'epoch': time.time(), 'rss_bytes': total, 'processes': records,
                                     'coverage': 'attached after build start; not complete historical peak'}) + '\n')
            stream.flush()
            if total > 16 * 1024**3 or any(record['rss_bytes'] > 8 * 1024**3 for record in records.values()):
                for pid in selected - {root_pid}:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                raise SystemExit('Bounded memory budget exceeded; compiler subtree TERM requested')
            time.sleep(0.5)
    print(json.dumps({'watch_exit': 0, 'peak_attached_rss_bytes': peak, 'not_complete_original_memory_proof': True}))


if __name__ == '__main__':
    main()

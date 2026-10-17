import argparse
import json
import os
from pathlib import Path
import re
import time


def snapshot():
    processes = []
    names = {'cc1', 'cc1plus', 'gcc', 'g++', 'clang', 'clang++', 'icpx', 'icx', 'ninja', 'make',
             'gmake', 'dkms', 'dpkg', 'apt', 'apt-get', 'ld', 'ld.lld', 'llvm-link', 'sycl-post-link'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / 'status').read_text()
            state = next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
            if state in ['Z', 'X']:
                continue
            name = (entry / 'comm').read_text().strip()
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            executable = Path(argv[0].decode(errors='replace')).name if argv and argv[0] else name
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        match = name in names or executable in names or any(re.fullmatch(r'(?:.*-)?(?:gcc|g\+\+|clang|clang\+\+)(?:-[0-9.]+)?', candidate) for candidate in [name, executable])
        if match:
            processes.append({'pid': int(entry.name), 'name': name, 'state': state, 'argv': [argument.decode(errors='replace') for argument in argv if argument]})
    stack = {'running_kernel': os.uname().release,
             'xe_version': Path('/sys/module/xe/version').read_text().strip(),
             'xe_srcversion': Path('/sys/module/xe/srcversion').read_text().strip()}
    return {'epoch': time.time(), 'external_or_native_compiler_processes': processes,
            'external_count': len(processes), 'stack': stack,
            'no_processes_stopped_or_modified': True, 'CPU_affinity': sorted(os.sched_getaffinity(0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--watch', action='store_true')
    options = parser.parse_args()
    directory = Path(__file__).resolve().parent
    assert sorted(os.sched_getaffinity(0)) == list(range(32, 48))
    initial = snapshot()
    if options.watch:
        accepted = json.loads((directory / 'CPU-maintenance-stack.json').read_text())['stack']
        while not (directory / 'maintenance-monitor-stop').exists():
            current = snapshot()
            with (directory / 'maintenance-during-performance.jsonl').open('a') as stream:
                stream.write(json.dumps(current) + '\n')
            if current['external_count'] or current['stack'] != accepted:
                (directory / 'performance-maintenance-hold.json').write_text(json.dumps(current, indent=2) + '\n')
                return
            time.sleep(2)
    else:
        Path(options.output).write_text(json.dumps(initial, indent=2) + '\n')
        print(json.dumps(initial, indent=2))
        raise SystemExit(125 if initial['external_count'] else 0)


if __name__ == '__main__':
    main()

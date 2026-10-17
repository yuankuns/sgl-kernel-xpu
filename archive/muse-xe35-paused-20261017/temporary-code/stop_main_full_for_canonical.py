import json
import os
import signal
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_PID = 70830
NINJA_PID = 71110


def processes():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(directory.name)] = {
                'ppid': int(fields[1]),
                'state': fields[0],
                'starttime': fields[19],
                'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
            }
        except (OSError, ValueError):
            continue
    return result


def live_original(captured):
    current = processes()
    return {
        process_id: item for process_id, item in captured.items()
        if process_id in current
        and current[process_id]['state'] != 'Z'
        and current[process_id]['starttime'] == item['starttime']
    }


def main():
    original = processes()
    if BUILD_PID not in original or not original[BUILD_PID]['command'].endswith('build-simple-container.sh '):
        raise SystemExit('Expected main build supervisor absent or changed; do not signal another process')
    if NINJA_PID not in original or 'ninja -j 8' not in original[NINJA_PID]['command']:
        raise SystemExit('Expected main Ninja absent or changed')
    captured = {BUILD_PID: original[BUILD_PID]}
    while True:
        descendants = {
            process_id: item for process_id, item in original.items()
            if item['ppid'] in captured and process_id not in captured
        }
        if not descendants:
            break
        captured.update(descendants)
    audit = {'captured': captured, 'signals': [], 'reason': 'coordinator canonical full-default pool transition'}
    os.kill(NINJA_PID, signal.SIGINT)
    audit['signals'].append({'pid': NINJA_PID, 'signal': 'SIGINT'})
    time.sleep(8)
    for signal_name in ('SIGTERM', 'SIGKILL'):
        remaining = live_original(captured)
        for process_id in remaining:
            try:
                os.kill(process_id, getattr(signal, signal_name))
                audit['signals'].append({'pid': process_id, 'signal': signal_name})
            except ProcessLookupError:
                pass
        time.sleep(3)
    audit['remaining_live'] = live_original(captured)
    (ROOT / 'simple-canonical-stop-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if audit['remaining_live']:
        raise SystemExit('Original main full subtree did not drain')
    print('Original main full subtree drained; audit preserved', flush=True)


if __name__ == '__main__':
    main()

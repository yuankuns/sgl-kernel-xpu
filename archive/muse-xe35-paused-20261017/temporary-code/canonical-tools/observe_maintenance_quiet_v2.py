import argparse
import json
import os
import time
from pathlib import Path

from build_pool_icpx_v2 import process_snapshot


PROCESS_NAMES = {'apt', 'apt-get', 'dpkg', 'dkms', 'make', 'ninja', 'gcc', 'g++',
                 'cc1', 'cc1plus', 'clang', 'clang++', 'icpx', 'icx'}


def loaded_stack():
    modules = {}
    for name in ('xe', 'i915'):
        directory = Path('/sys/module') / name
        if directory.exists():
            modules[name] = {field: (directory / field).read_text().strip()
                             for field in ('version', 'srcversion', 'initstate')
                             if (directory / field).is_file()}
    return {'running_kernel': os.uname().release, 'loaded_modules': modules}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite observation evidence')
    expected = loaded_stack()
    assert expected['running_kernel'] == '7.0.0-38-generic'
    assert expected['loaded_modules']['xe']['srcversion'] == '858285FCD284B7EBB0E410F'
    start = time.monotonic()
    samples = []
    status = 0
    while True:
        processes = process_snapshot()
        active = {pid: record for pid, record in processes.items() if record['name'] in PROCESS_NAMES}
        current = loaded_stack()
        samples.append({'epoch': time.time(), 'active_processes': active, 'loaded_stack': current})
        if active or current != expected:
            status = 1
            break
        if time.monotonic() - start >= 60:
            break
        time.sleep(2)
    report = {'actual_exit': status, 'samples': samples, 'continuous_quiet_seconds': time.monotonic() - start,
              'loaded_stack': expected, 'GPU_work_started': False, 'compiler_started': False,
              'external_processes_stopped_or_modified': False, 'new_grant_automatically_issued': False,
              'scope': 'Non-zombie process-name scans and loaded stack over one bounded 60-second interval; final prelaunch rescan remains required'}
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'samples'}))
    raise SystemExit(status)


if __name__ == '__main__':
    main()

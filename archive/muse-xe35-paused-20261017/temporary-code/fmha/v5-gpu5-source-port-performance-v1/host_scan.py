import fcntl
import json
import os
from pathlib import Path
import time


COMPILERS = {'icpx', 'icx', 'clang', 'clang++', 'clang-21', 'clang-22', 'gcc', 'g++', 'cc', 'c++',
    'cc1', 'cc1plus', 'ninja', 'make', 'cmake', 'sycl-post-link', 'llvm-spirv', 'ld.lld', 'ld', 'lld', 'aoc'}
PACKAGE_MANAGERS = {'dpkg', 'dpkg-deb', 'apt', 'apt-get', 'dkms', 'pip', 'pip3', 'rpm', 'modprobe', 'insmod', 'rmmod'}


def snapshot(probe_lock=True):
    compilers, managers = {}, {}
    terminal_zombies = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            words = (directory / 'cmdline').read_bytes().split(b'\0')
            command = [word.decode(errors='replace') for word in words if word]
            name = (directory / 'comm').read_text().strip()
            state = (directory / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        executable = Path(command[0]).name if command else name
        record = {'name': name, 'command': command, 'state': state}
        if state in ('Z', 'X'):
            if name in COMPILERS or executable in COMPILERS or name in PACKAGE_MANAGERS:
                terminal_zombies[directory.name] = record
            continue
        if name in COMPILERS or executable in COMPILERS:
            compilers[directory.name] = record
        if name in PACKAGE_MANAGERS or executable in PACKAGE_MANAGERS or any(command[index:index + 2] == ['-m', 'pip'] for index in range(len(command))):
            managers[directory.name] = record
    assert not compilers and not managers, (compilers, managers)
    modules = {}
    for name in ('xe', 'i915'):
        directory = Path('/sys/module') / name
        if directory.exists():
            modules[name] = {field: (directory / field).read_text().strip()
                for field in ('version', 'srcversion', 'initstate') if (directory / field).is_file()}
    report = {'epoch': time.time(), 'actual_exit': 0, 'matching_live_compiler_processes': compilers,
        'matching_live_package_manager_processes': managers,
        'terminal_compiler_zombies_not_live': terminal_zombies,
        'loaded_stack': {'running_kernel': os.uname().release, 'loaded_modules': modules}}
    main = Path(__file__).resolve().parent.parent.parent
    earlier = json.loads((main / 'PERFORMANCE_POOL_ROUTING_POST_DRAIN_v1.json').read_text())
    assert report['loaded_stack'] == earlier['loaded_stack']
    if probe_lock:
        with Path('/tmp/intel-gpu-5.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        report['GPU5_actual_flock_free'] = True
    return report


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))

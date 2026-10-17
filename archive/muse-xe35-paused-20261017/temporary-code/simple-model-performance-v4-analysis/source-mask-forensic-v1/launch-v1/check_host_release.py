import hashlib
import json
import subprocess
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
hosts = (3328794, 3328795, 3328965, 3328994)
assert (DIRECTORY / 'wrapper.exit').read_text().strip() == '1'
assert all(not Path('/proc', str(pid)).exists() for pid in hosts)
lock_exit = subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true']).returncode
assert lock_exit == 0
record = {'actual_host_release_check_exit': 0, 'actual_wrapper_exit': 1, 'epoch': time.time(), 'host_PIDs_absent': list(hosts), 'GPU7_physical_lock_probe_exit': lock_exit, 'no_GPU_restarted': True, 'no_performance_started': True, 'grant_sha256': hashlib.sha256((DIRECTORY / 'main-grant.json').read_bytes()).hexdigest()}
with (DIRECTORY / 'HOST_RELEASE.json').open('x') as output:
    output.write(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))

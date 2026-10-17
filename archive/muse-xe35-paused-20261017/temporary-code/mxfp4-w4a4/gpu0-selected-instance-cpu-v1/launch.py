import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from protocol import DIRECTORY, digest, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.parse_args()
    grant_path = DIRECTORY / 'main-gpu0-diagnostic-grant.json'
    if not grant_path.exists():
        parser.error('new explicit main GPU0 diagnostic grant required; no GPU lock/init started')
    assert not (DIRECTORY / 'host-supervisor.json').exists(), 'once only; no retry'
    ready_path = DIRECTORY / 'CPU-READY.json'
    ready = json.loads(ready_path.read_text())
    grant = json.loads(grant_path.read_text())
    assert ready['CPU_READY'] and grant['issuer'] == 'main' and grant['release']
    assert grant['gpu'] == 0 and grant['diagnostic_only'] and not grant['performance_authorized']
    assert grant['fresh_GPU0_owned_drain_and_lockfree_verified'] is True
    assert grant['CPU_READY_SHA256'] == digest(ready_path)
    assert grant['bound_files_SHA256'] == ready['bound_files_SHA256']
    assert grant['start_epoch'] >= grant['grant_epoch'] + 120
    assert len(set(grant['CPU_affinity'])) == len(grant['CPU_affinity']) and 1 <= len(grant['CPU_affinity']) <= 16
    assert not set(grant['CPU_affinity']) & set(range(64, 80)), 'do not overlap Routing8 live CPU64-79'
    assert grant['OMP_NUM_THREADS'] == 8
    save(DIRECTORY / 'host-supervisor.json', {'PID': os.getpid(), 'grant_epoch': grant['grant_epoch'], 'start_epoch': grant['start_epoch']})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    for name, expected in ready['bound_files_SHA256'].items():
        if not name.startswith('/opt/'):
            assert digest(name) == expected, name
    container_directory = str(DIRECTORY).replace('/home/syk/', '/root/', 1)
    cpus = ','.join(map(str, grant['CPU_affinity']))
    command = ['/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh', '--gpu-lock', '--gpu', '0',
               '--workdir', container_directory, '--', 'bash', '-c',
               'directory="$1"; cpus="$2"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1; exec taskset -c "$cpus" /opt/gfx-deps/venv/bin/python "$directory/run_supervisor.py"',
               'bash', container_directory, cpus]
    environment = os.environ.copy()
    environment['SGLANG_MUSE_CONTAINER'] = 'codex-muse-bf16-review-20261015'
    with (DIRECTORY / 'wrapper.log').open('xb') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    (DIRECTORY / 'host-launch.exit').write_text(str(result.returncode) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

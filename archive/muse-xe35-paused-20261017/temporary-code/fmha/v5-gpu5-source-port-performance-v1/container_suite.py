import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    grant_path = ROOT / 'gpu5-source-port-barrier-v1.json'
    grant = json.loads(grant_path.read_text())
    assert grant['container_supervisor_sha256'] == digest(Path(__file__))
    assert grant['phase'] == 'fmha-v5-gpu5-source-port-all89-three-BAAB'
    assert os.environ['ZE_AFFINITY_MASK'] == '5' and os.sched_getaffinity(0) == set(range(16, 32))
    assert time.time() >= grant['epoch']
    with (ROOT / 'container-suite-start.json').open('x') as stream:
        json.dump({'runtime_container_pid': os.getpid(), 'epoch': time.time(),
            'grant_sha256': digest(grant_path), 'one_wrapper_GPU5_flock_for_all12_slots': True}, stream, indent=2)
    print(json.dumps({'container_supervisor_pid': os.getpid(), 'slot_order': 'BAAB', 'rounds': 3}), flush=True)
    runtime = ART / 'v5-gpu5-performance-preparation/performance_driver.py'
    slots = []
    for round_number in (1, 2, 3):
        for slot in ('B0', 'A1', 'A2', 'B3'):
            assert digest(runtime) == grant['driver_sha256']
            role = 'port' if slot.startswith('B') else 'source'
            name = f'source-port-r{round_number}-{slot}'
            assert not (ROOT / f'{name}.json').exists() and not (ROOT / f'{name}.handle.json').exists()
            site = ART / ('canonical-v5-icpx/site' if role == 'port' else 'matched-native-icpx-v1/source/site')
            environment = dict(os.environ)
            environment['FMHA_CANONICAL_VERSION'] = 'canonical-v5-icpx'
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            environment['PYTHONPATH'] = f'{site}:/root/sglang/python:/root/sglang/sglang_995c774/sglang_meta_common'
            environment['LD_LIBRARY_PATH'] = f'/opt/gfx-deps/profiling_tools/lib:{site}/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
            command = [sys.executable, str(runtime), '--comparison', 'source-port', '--role', role,
                '--round', str(round_number), '--slot', slot, '--grant', str(grant_path), '--output', str(ROOT / f'{name}.json')]
            with (ROOT / f'{name}.log').open('x') as log:
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                handle = {'runtime_container_pid': child.pid, 'supervisor_container_pid': os.getpid(),
                    'epoch': time.time(), 'command': command, 'role': role, 'round': round_number, 'slot': slot}
                with (ROOT / f'{name}.handle.json').open('x') as stream:
                    json.dump(handle, stream, indent=2)
                print(json.dumps(handle), flush=True)
                status = child.wait()
            with (ROOT / f'{name}.exit').open('x') as stream:
                stream.write(f'{status}\n')
            slots.append({**handle, 'actual_exit': status, 'terminal_epoch': time.time()})
            with (ROOT / 'suite-progress.json').open('w') as stream:
                json.dump({'slots': slots, 'stopped_on_first_failure': status != 0, 'auto_retry': False}, stream, indent=2)
            print(json.dumps({'slot_terminal': name, 'actual_exit': status}), flush=True)
            if status != 0:
                raise SystemExit(status if status > 0 else 128 - status)
    print(json.dumps({'all12_slots_actual_exit': 0, 'analysis': 'host CPU after physical GPU wrapper drain'}), flush=True)


if __name__ == '__main__':
    main()

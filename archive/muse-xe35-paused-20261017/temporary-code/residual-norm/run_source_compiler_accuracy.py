import json
import os
from pathlib import Path
import subprocess
import sys
import time

import source_compiler_accuracy as check


def command(name, argv):
    output = check.OUTPUT
    (output / f'{name}.command.json').write_text(json.dumps(argv, indent=2) + '\n')
    with (output / f'{name}.log').open('w') as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (output / f'{name}.pid').write_text(str(process.pid) + '\n')
        status = process.wait()
    (output / f'{name}.exit').write_text(str(status) + '\n')
    if status != 0:
        raise subprocess.CalledProcessError(status, argv)


def main():
    check.scope()
    if len(sys.argv) != 1:
        raise ValueError('no arguments accepted; accuracy-only B8 scope')
    check.OUTPUT.mkdir()
    (check.OUTPUT / 'host-supervisor.pid').write_text(str(os.getpid()) + '\n')
    record = check.audit()
    (check.OUTPUT / 'cpu-preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    container_root = '/root/' + str(check.ROOT).removeprefix('/home/syk/')
    os.environ['SGLANG_MUSE_CONTAINER'] = 'codex-muse-bf16-review-20261015'
    wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
    script = container_root + '/source_compiler_accuracy_container.sh'
    command('CPU-container-preflight', [wrapper, '--workdir', container_root, 'taskset', '-c', '48-63', 'bash', script, 'preflight'])
    scripts = {name: check.digest(check.ROOT / name) for name in
               ['source_compiler_accuracy.py', 'source_compiler_accuracy_container.sh', 'run_source_compiler_accuracy.py']}
    ready = dict(record, host_pid=os.getpid(), ready_epoch=time.time(), gpu=7, CPU='48-63',
                 barrier_path=str(check.BARRIER), phase=check.PHASE, status='READY waiting fresh main grant',
                 script_sha256=scripts, GPU_started=False, scope='B8/ISL4096/OSL16/TP1/EP1')
    (check.OUTPUT / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready, indent=2), flush=True)
    while True:
        try:
            released = check.grant()
            if time.time() >= released['start_epoch']:
                break
        except (OSError, ValueError, KeyError, AssertionError) as error:
            (check.OUTPUT / 'wait-status.json').write_text(json.dumps({'epoch': time.time(), 'waiting': str(error)}) + '\n')
        time.sleep(2)
    assert all(check.digest(check.ROOT / name) == value for name, value in scripts.items())
    assert check.audit() == record
    (check.OUTPUT / 'consumed-grant.json').write_text(json.dumps(released, indent=2) + '\n')
    command('GPU7-wrapper', [wrapper, '--workdir', container_root, '--gpu-lock', '--gpu', '7',
                             'taskset', '-c', '48-63', 'bash', script, 'runtime'])


if __name__ == '__main__':
    status = 0
    try:
        main()
    except subprocess.CalledProcessError as error:
        status = error.returncode if error.returncode > 0 else 128 - error.returncode
        raise
    except BaseException:
        status = 1
        raise
    finally:
        if check.OUTPUT.is_dir():
            (check.OUTPUT / 'supervisor.exit').write_text(str(status) + '\n')
            (check.OUTPUT / 'supervisor-terminal.json').write_text(json.dumps(
                {'epoch': time.time(), 'actual_exit': status, 'host_pid': os.getpid(),
                 'official_AFTER': False, 'accuracy_only': True}) + '\n')

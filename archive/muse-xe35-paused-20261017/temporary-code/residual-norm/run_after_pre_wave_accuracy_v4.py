import json
import os
from pathlib import Path
import subprocess
import sys
import time

import after_pre_wave_accuracy_v4 as check


OWN_OUTPUT = False


def command(name, argv):
    (check.OUTPUT / f'{name}.command.json').write_text(json.dumps(argv, indent=2) + '\n')
    with (check.OUTPUT / f'{name}.log').open('w') as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / f'{name}.pid').write_text(str(process.pid) + '\n')
        status = process.wait()
    (check.OUTPUT / f'{name}.exit').write_text(str(status) + '\n')
    if status:
        raise subprocess.CalledProcessError(status, argv)


def main():
    global OWN_OUTPUT
    check.source_check.scope()
    if len(sys.argv) != 1:
        raise ValueError('accuracy-only entry accepts no timing/scope overrides')
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    assert not check.BARRIER.exists()
    check.OUTPUT.mkdir()
    OWN_OUTPUT = True
    (check.OUTPUT / 'host-supervisor.pid').write_text(str(os.getpid()) + '\n')
    os.environ['SGLANG_MUSE_CONTAINER'] = 'codex-muse-bf16-review-20261015'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    container = '/root/' + str(check.ROOT).removeprefix('/home/syk/')
    wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
    script = container + '/after_pre_wave_accuracy_v4_container.sh'
    command('CPU-wrapper', [wrapper, '--workdir', container, 'taskset', '-c', '48-63', 'bash', script, 'preflight'])
    assert (check.OUTPUT / 'CPU-provider-protected-source.exit').read_text().strip() == '0'
    record = json.loads((check.OUTPUT / 'CPU-preflight.json').read_text())
    scripts = {name: check.digest(check.ROOT / name) for name in
               ['after_pre_wave_accuracy_v4.py', 'after_pre_wave_accuracy_v4_container.sh', 'run_after_pre_wave_accuracy_v4.py',
                'source_compiler_accuracy.py', 'check_canonical_package_v3.py']}
    ready = dict(record, host_pid=os.getpid(), ready_epoch=time.time(), GPU_started=False, script_sha256=scripts)
    (check.OUTPUT / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true'], check=True)
    released = dict(record, issuer='main coordinator user authorization', gpu=7, CPU='48-63',
                    phase=check.PHASE, coordinator_release=True, accuracy_only=True,
                    scope='B8/ISL4096/OSL16/TP1/EP1', start_epoch=time.time() + 10,
                    no_timing_events_graphs_warmup=True, no_limited_PORT=True,
                    formal_perf_barrier_consumed=False, other_full_compilers_may_remain_live=True,
                    script_sha256=scripts)
    patch = '*** Begin Patch\n*** Add File: ' + str(check.BARRIER) + '\n'
    patch += ''.join('+' + line + '\n' for line in json.dumps(released, indent=2).splitlines())
    subprocess.run(['apply_patch', patch + '*** End Patch\n'], check=True)
    print(json.dumps({'grant': released, 'host_pid': os.getpid(), 'status': 'CPU0; fresh grant future epoch'}, indent=2), flush=True)
    while time.time() < released['start_epoch']:
        time.sleep(1)
    assert check.grant() == released
    assert all(check.digest(check.ROOT / name) == value for name, value in scripts.items())
    (check.OUTPUT / 'consumed-grant.json').write_text(json.dumps(released, indent=2) + '\n')
    command('GPU7-wrapper', [wrapper, '--workdir', container, '--gpu-lock', '--gpu', '7',
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
        if OWN_OUTPUT:
            (check.OUTPUT / 'supervisor.exit').write_text(str(status) + '\n')
            (check.OUTPUT / 'supervisor-terminal.json').write_text(json.dumps(
                {'epoch': time.time(), 'actual_exit': status, 'host_pid': os.getpid(),
                 'accuracy_only': True, 'performance_approved': False}) + '\n')

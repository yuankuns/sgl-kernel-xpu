import fcntl
import json
import os
import shlex
import subprocess
import time
import traceback
from pathlib import Path

from source_accuracy_package_check import preflight


def main():
    os.umask(0o022)
    root = Path(__file__).resolve().parent
    stage = root / 'source-native-icpx-v1-accuracy'
    gpu = 4 if root.name == 'bf16-gemm' else 10
    affinity = '0-15' if root.name == 'bf16-gemm' else '96-111'
    lock = (root / '.canonical-official-supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (stage / 'supervisor.started.json').exists():
        raise SystemExit('Refusing duplicate SOURCE accuracy supervisor')
    grant = stage / 'fresh-grant.json'
    preflight(root, grant, gpu)
    (stage / 'supervisor.started.json').write_text(json.dumps({'pid': os.getpid(), 'epoch': time.time(),
                                                             'gpu': gpu, 'scope': 'SOURCE accuracy only'}) + '\n')
    container_root = str(root).replace('/home/syk/', '/root/', 1)
    wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
    prefix = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015', wrapper]
    runtime = prefix + ['--gpu-lock', '--gpu', str(gpu), '--workdir', container_root, '--',
              'bash', container_root + '/source_accuracy_container.sh']
    status = 1
    try:
        (stage / 'GPU-round.command').write_text(shlex.join(runtime) + '\n')
        with (stage / 'GPU-round.log').open('w') as output:
            child = subprocess.Popen(runtime, stdout=output, stderr=subprocess.STDOUT)
            (stage / 'GPU-wrapper.started.json').write_text(json.dumps({'pid': child.pid, 'epoch': time.time()}) + '\n')
            status = child.wait()
        (stage / 'GPU-round.exit').write_text(str(status) + '\n')
        (stage / 'GPU-drained.json').write_text(json.dumps({'gpu': gpu, 'wrapper_returned': True, 'epoch': time.time(),
                                                          'actual_exit': status, 'lock_lifetime_ended': True}) + '\n')
        if status:
            raise RuntimeError('SOURCE accuracy GPU round failed: ' + str(status))
        analyze = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                  'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                  'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                  'exec taskset -c "$2" python -B "$1/source_accuracy_analyze.py"', '_', container_root, affinity]
        (stage / 'analyze.command').write_text(shlex.join(analyze) + '\n')
        with (stage / 'analyze.log').open('w') as output:
            status = subprocess.run(analyze, stdout=output, stderr=subprocess.STDOUT).returncode
        (stage / 'analyze.exit').write_text(str(status) + '\n')
    except Exception:
        status = status or 1
        traceback.print_exc()
    (stage / 'supervisor.exit').write_text(str(status) + '\n')
    print(json.dumps({'actual_exit': status, 'gpu': gpu, 'performance_collected': False,
                      'AFTER_tested': False, 'stage': str(stage)}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()

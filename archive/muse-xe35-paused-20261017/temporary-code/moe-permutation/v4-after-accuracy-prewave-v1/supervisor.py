import fcntl
import json
import os
import shlex
import subprocess
import time
import traceback

from checks import ROOT, ROUND, SCOPE, VERSION, digest, driver_hashes, grant_gate, package_gate, write_json


def run(name, command):
    (ROUND / (name + '.command')).write_text(shlex.join(command) + '\n')
    with (ROUND / (name + '.log')).open('w') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        write_json(ROUND / (name + '.started.json'), {'host_wrapper_pid': child.pid, 'epoch': time.time(), 'command': command})
        status = child.wait()
    (ROUND / (name + '.exit')).write_text(str(status) + '\n')
    if name == 'GPU-round':
        write_json(ROUND / 'GPU-drained.json', {'gpu': 10, 'epoch': time.time(), 'wrapper_returned': True,
                                             'actual_exit': status, 'GPU_lock_lifetime_ended': True})
    if status:
        raise RuntimeError(name + ' actual exit=' + str(status))


def main():
    os.umask(0o022)
    if os.environ.get('BATCHES', '8') != '8':
        raise SystemExit('B8 only before GPU lock/import')
    if (ROUND / 'supervisor.started.json').exists():
        raise SystemExit('Refusing duplicate pre-wave supervisor')
    lock = (ROOT / '.canonical-official-supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(ROUND / 'supervisor.started.json', {'pid': os.getpid(), 'epoch': time.time(), 'scope': SCOPE,
                                               'performance_authorized': False, 'formal_perf_barrier_consumed': False})
    status = 1
    try:
        manifest, manifest_file, site, protected = package_gate()
        container_root = str(ROOT).replace('/home/syk/', '/root/', 1)
        container_round = str(ROUND).replace('/home/syk/', '/root/', 1)
        wrapper = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'
        prefix = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015', wrapper]
        CPU = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                        'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                        'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                        'export PYTHONPATH="$1/canonical-v4-icpx/site"; '
                        'exec taskset -c 96-111 python -B "$2/cpu_gate.py"', '_', container_root, container_round]
        run('cpu-gate', CPU)
        cpu_gate = json.loads((ROUND / 'cpu-normal-package-gate.json').read_text())
        files, identity = driver_hashes()
        epoch = time.time() + 5
        grant = {'version': VERSION, 'scope': SCOPE, 'module': 'moe-permutation', 'gpu': 10,
                 'accuracy_only': True, 'performance_authorized': False, 'formal_perf_barrier_consumed': False,
                 'coordinator_release': True, 'authorization': 'main explicit independent pre-wave accuracy grant, not formal eight-card wave',
                 'epoch': epoch, 'not_before_epoch': epoch, 'CPU_affinity': '96-111',
                 'after_manifest_sha256': digest(manifest_file), 'source_digest': protected['source_digest'],
                 'source_freeze_sha256': protected['freeze_sha256'],
                 'compiler_policy_sha256': digest(ROOT / VERSION / 'compiler-policy-verified.json'),
                 'source_native_manifest_sha256': digest(ROOT / 'fresh-source-icpx-v1/source-native-binary-manifest.json'),
                 'driver_sha256': identity, 'driver_files_sha256': files,
                 'cpu_gate_sha256': digest(ROUND / 'cpu-normal-package-gate.json'),
                 'shared_SOURCE_inputs_sha256': digest(ROUND / 'shared-SOURCE-inputs.pt')}
        write_json(ROUND / 'fresh-grant.json', grant)
        grant_gate()
        command = prefix + ['--gpu-lock', '--gpu', '10', '--workdir', container_root,
                            '--', 'bash', container_round + '/gpu_container.sh']
        run('GPU-round', command)
        analyzer = prefix + ['--workdir', container_root, '--', 'bash', '-c',
                             'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                             'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                             'exec taskset -c 96-111 python -B "$1/analyze.py"', '_', container_round]
        run('analyze', analyzer)
        package_gate()
        status = 0
    except Exception:
        traceback.print_exc()
    (ROUND / 'supervisor.exit').write_text(str(status) + '\n')
    print(json.dumps({'supervisor_actual_exit': status, 'scope': SCOPE, 'performance_collected': False,
                      'formal_perf_barrier_consumed': False, 'artifact': str(ROUND)}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()

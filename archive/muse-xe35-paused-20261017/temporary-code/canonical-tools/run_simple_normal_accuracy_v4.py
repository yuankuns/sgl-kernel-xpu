import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = Path('/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def container_path(path):
    return '/root/' + str(path.relative_to('/home/syk'))


def main():
    run = ROOT / 'simple-normal-accuracy-v4-run1'
    run.mkdir(exist_ok=False)
    start = time.time() + 120
    validator = ROOT / 'validate_simple_canonical_v1.py'
    drivers = {
        name: digest(ROOT / name)
        for name in ('validate_simple_canonical_v1.py', 'benchmark_simple_canonical_v1.py',
                     'benchmark_simple.py', 'check_simple_canonical_package_v1.py',
                     'canonical_tools_loader_v1.py', 'acceptance_cooldown.py')
    }
    commands = []
    for component in ('memory', 'position', 'logits-vocab'):
        official = ROOT / component / 'canonical-v4-icpx'
        evidence = ROOT / component / 'canonical-validation-v4-icpx'
        for label in ('before', 'after'):
            if (evidence / f'CPU-normal-{label}.exit').read_text().strip() != '0':
                raise RuntimeError(f'CPU normal-package preflight failed: {component}/{label}')
        snapshot = json.loads((official / 'source-snapshot.json').read_text())
        comparator_path = evidence / 'source-comparator-manifest.json'
        comparator = json.loads(comparator_path.read_text())
        grant = {
            'version': 'canonical-v4-icpx', 'module': component, 'gpu': 11,
            'phase': 'accuracy', 'coordinator_release': True,
            'source_digest': snapshot['source_digest'],
            'compiler_policy_sha256': digest(official / 'compiler-policy-verified.json'),
            'package_manifest_sha256': digest(official / 'source-binary-manifest.json'),
            'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
            'source_comparator_manifest_sha256': digest(comparator_path),
            'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
            'accuracy_authorized': True, 'performance_authorized': False,
            'start_epoch': start, 'issuer': 'main-coordinator',
            'driver_sha256': drivers, 'container_cpu_affinity': '112-127',
            'scope': 'original tests and representative kernel contracts; not whole-model acceptance',
        }
        grant_path = run / f'{component}-grant.json'
        grant_path.write_text(json.dumps(grant, indent=2) + '\n')
        command = [WRAPPER.as_posix(), '--gpu-lock', '--gpu', '11', '--workdir',
                   container_path(ROOT), '--', 'bash', '-c',
                   'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                   'export PYTHONDONTWRITEBYTECODE=1; '
                   'exec taskset -c 112-127 /opt/gfx-deps/venv/bin/python "$@"',
                   'bash', container_path(validator), component, '--phase', 'accuracy',
                   '--grant', container_path(grant_path)]
        commands.append((component, command))
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    (run / 'launch-manifest.json').write_text(json.dumps({
        'host_pid': os.getpid(), 'start_epoch': start, 'gpu': 11,
        'driver_sha256': drivers, 'launcher_sha256': digest(Path(__file__)),
        'commands': {component: command for component, command in commands},
    }, indent=2) + '\n')
    for component, command in commands:
        if any(digest(ROOT / name) != expected for name, expected in drivers.items()):
            raise RuntimeError('Frozen validation driver changed')
        (run / f'{component}.command').write_text(shlex.join(command) + '\n')
        with (run / f'{component}.log').open('x') as output:
            worker = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
            (run / f'{component}.wrapper-pid').write_text(str(worker.pid) + '\n')
            actual_exit = worker.wait()
        (run / f'{component}.exit').write_text(str(actual_exit) + '\n')
        print(f'{component}: actual wrapper exit {actual_exit}', flush=True)
        if actual_exit:
            raise SystemExit(actual_exit)
    (run / 'complete.json').write_text(json.dumps({'actual_exit': 0, 'components': 3,
                                                 'precision_only': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()

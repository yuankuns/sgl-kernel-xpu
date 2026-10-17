import ast
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def container_path(path):
    return '/root/' + str(path.relative_to('/home/syk'))


def execute(command, environment, prefix):
    prefix.with_suffix('.command').write_text(shlex.join(command) + '\n')
    with prefix.with_suffix('.log').open('x') as output:
        worker = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
        prefix.with_suffix('.pid').write_text(str(worker.pid) + '\n')
        result = worker.wait()
    prefix.with_suffix('.exit').write_text(str(result) + '\n')
    print(f'{prefix.name}: actual exit {result}', flush=True)
    if result:
        raise SystemExit(result)


def main():
    run = ROOT / 'logits-model-config-accuracy-v2-run1'
    run.mkdir(exist_ok=False)
    predecessor = ROOT / 'simple-normal-accuracy-v4-run1'
    launch = json.loads((predecessor / 'launch-manifest.json').read_text())
    predecessor_pid = launch['host_pid']
    stat_path = Path(f'/proc/{predecessor_pid}/stat')
    start_tick = stat_path.read_text().split(') ')[1].split()[19] if stat_path.exists() else None
    driver = ROOT / 'benchmark_logits_model_config_v2.py'
    auditor = ROOT / 'canonical-tools/audit_logits_model_config_v2.py'
    for path in (driver, auditor):
        ast.parse(path.read_text())
    driver_sha, auditor_sha = digest(driver), digest(auditor)
    preflight = json.loads((ROOT / 'logits-model-config-v2-cpu-preflight.json').read_text())
    assert preflight['verified'] and preflight['driver_sha256'] == driver_sha
    (run / 'conditional-main-authorization.json').write_text(json.dumps({
        'issuer': 'main-coordinator', 'gpu': 11, 'phase': 'accuracy',
        'predecessor': str(predecessor), 'predecessor_pid': predecessor_pid,
        'predecessor_start_tick': start_tick, 'host_pid': os.getpid(),
        'driver_sha256': driver_sha, 'auditor_sha256': auditor_sha,
        'condition': 'all three original/representative normal-package accuracy runs actual0 and predecessor drained',
        'CPU_affinity': '112-127', 'performance_authorized': False,
    }, indent=2) + '\n')
    while stat_path.exists():
        if stat_path.read_text().split(') ')[1].split()[19] != start_tick:
            break
        time.sleep(10)
    assert json.loads((predecessor / 'complete.json').read_text())['actual_exit'] == 0
    for component in ('memory', 'position', 'logits-vocab'):
        assert (predecessor / f'{component}.exit').read_text().strip() == '0'
        evidence = ROOT / component / 'canonical-validation-v4-icpx'
        for label in ('before', 'after'):
            for phase in ('original', 'production'):
                assert (evidence / f'accuracy-{phase}-{label}.exit').read_text().strip() == '0'
    assert digest(driver) == driver_sha and digest(auditor) == auditor_sha
    official = ROOT / 'logits-vocab/canonical-v4-icpx'
    comparator_path = ROOT / 'logits-vocab/canonical-validation-v4-icpx/source-comparator-manifest.json'
    comparator = json.loads(comparator_path.read_text())
    provenance = json.loads((ROOT / 'SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json').read_text())
    assert digest(Path(provenance['model_config'])) == provenance['model_config_sha256']
    grant = {'issuer': 'main-coordinator', 'scope': 'logits-model-config-v2',
             'phase': 'accuracy', 'gpu': 11, 'serving_batch': 8, 'isl': 4096,
             'osl': 16, 'tp': 1, 'ep': 1, 'coordinator_release': True,
             'accuracy_authorized': True, 'performance_authorized': False,
             'driver_sha256': driver_sha, 'model_config_sha256': provenance['model_config_sha256'],
             'package_manifest_sha256': digest(official / 'source-binary-manifest.json'),
             'source_comparator_manifest_sha256': digest(comparator_path),
             'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
             'source_digest': json.loads((official / 'source-snapshot.json').read_text())['source_digest'],
             'compiler_policy_sha256': digest(official / 'compiler-policy-verified.json'),
             'start_epoch': time.time() + 120}
    grant_path = run / 'fresh-grant.json'
    grant_path.write_text(json.dumps(grant, indent=2) + '\n')
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    for label in ('before', 'after'):
        stage = official / 'site' if label == 'after' else Path(comparator['stage'].replace('/root/', '/home/syk/', 1))
        python_path = ':'.join((container_path(stage), container_path(ROOT / 'logits-vocab/framework-original/site'), container_path(ROOT)))
        command = [WRAPPER, '--gpu-lock', '--gpu', '11', '--workdir', container_path(ROOT), '--',
                   'bash', '-c',
                   'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                   'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; '
                   'export PYTHONPATH="$1"; shift; '
                   'exec taskset -c 112-127 /opt/gfx-deps/venv/bin/python "$@"',
                   'bash', python_path, container_path(driver), '--label', label, '--phase', 'accuracy',
                   '--grant', container_path(grant_path), '--output', container_path(run / f'{label}.json')]
        execute(command, environment, run / label)
    command = [WRAPPER, '--workdir', container_path(ROOT), '--', 'bash', '-c',
               'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
               'export PYTHONDONTWRITEBYTECODE=1; '
               'exec taskset -c 112-127 /opt/gfx-deps/venv/bin/python "$@"',
               'bash', container_path(auditor), container_path(run)]
    execute(command, environment, run / 'bitwise-audit')
    (run / 'complete.json').write_text(json.dumps({'actual_exit': 0, 'GPU': 11,
                                                 'performance_authorized': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()

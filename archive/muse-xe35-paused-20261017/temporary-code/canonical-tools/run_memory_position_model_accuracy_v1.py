import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def container_path(path):
    return '/root/' + str(path.relative_to('/home/syk'))


def main():
    run = ROOT / 'memory-position-model-config-accuracy-v1-run1'
    run.mkdir(exist_ok=False)
    driver = ROOT / 'benchmark_memory_position_model_config_v1.py'
    ast.parse(driver.read_text())
    assert json.loads((ROOT / 'SIMPLE_NORMAL_ICPX_ACCURACY_AUDIT_v4.json').read_text())['actual_exit'] == 0
    assert json.loads((ROOT / 'logits-model-config-accuracy-v2-run1/complete.json').read_text())['actual_exit'] == 0
    aliases = ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE',
               'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS')
    guard_results = []
    for component in ('memory', 'position'):
        tests = [({'ZE_AFFINITY_MASK': '0'}, 'Accuracy11/performance7'),
                 ({'ZE_AFFINITY_MASK': '8'}, 'Accuracy11/performance7')]
        tests += [({'ZE_AFFINITY_MASK': '11', name: '16'}, 'B8-only') for name in aliases]
        tests += [({'ZE_AFFINITY_MASK': '11', key: value}, 'Wrong production scope')
                  for key, value in {'ISL': '2048', 'OSL': '32', 'TP': '2', 'EP': '2'}.items()]
        command = [sys.executable, '-S', str(driver), component, '--label', 'before',
                   '--grant', '/nonexistent-main-grant', '--output', '/nonexistent-main-output']
        for overrides, expected in tests:
            environment = dict(os.environ)
            for key in (*aliases, 'ISL', 'OSL', 'TP', 'EP', 'ZE_AFFINITY_MASK'):
                environment.pop(key, None)
            environment.update(overrides)
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            assert result.returncode != 0 and expected in result.stderr
            assert 'No module named' not in result.stderr
            guard_results.append({'component': component, 'overrides': overrides,
                                  'actual_exit': result.returncode, 'stderr': result.stderr})
    driver_sha = digest(driver)
    (run / 'CPU-guards.json').write_text(json.dumps({'actual_exit': 0, 'driver_sha256': driver_sha,
                                                   'negative_guards': guard_results, 'GPU_started': False}, indent=2) + '\n')
    provenance = json.loads((ROOT / 'SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json').read_text())
    assert digest(Path(provenance['model_config'])) == provenance['model_config_sha256']
    start_epoch = time.time() + 120
    (run / 'launch-state.json').write_text(json.dumps({'host_pid': os.getpid(), 'gpu': 11,
                                                      'container_cpu_affinity': '112-127',
                                                      'start_epoch': start_epoch}, indent=2) + '\n')
    environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015')
    for component in ('memory', 'position'):
        official = ROOT / component / 'canonical-v4-icpx'
        comparator_path = ROOT / component / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
        comparator = json.loads(comparator_path.read_text())
        grant = {'scope': 'memory-position-model-config-v1', 'module': component, 'issuer': 'main-coordinator',
                 'phase': 'accuracy', 'gpu': 11, 'coordinator_release': True,
                 'accuracy_authorized': True, 'performance_authorized': False,
                 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
                 'driver_sha256': driver_sha, 'model_config_sha256': provenance['model_config_sha256'],
                 'source_digest': json.loads((official / 'source-snapshot.json').read_text())['source_digest'],
                 'compiler_policy_sha256': digest(official / 'compiler-policy-verified.json'),
                 'package_manifest_sha256': digest(official / 'source-binary-manifest.json'),
                 'source_comparator_manifest_sha256': digest(comparator_path),
                 'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
                 'start_epoch': start_epoch}
        grant_path = run / f'{component}-grant.json'
        grant_path.write_text(json.dumps(grant, indent=2) + '\n')
        for label in ('before', 'after'):
            assert digest(driver) == driver_sha
            stage = official / 'site' if label == 'after' else Path(comparator['stage'].replace('/root/', '/home/syk/', 1))
            python_path = ':'.join((container_path(stage), container_path(ROOT / 'logits-vocab/framework-original/site'), container_path(ROOT)))
            command = [WRAPPER, '--gpu-lock', '--gpu', '11', '--workdir', container_path(ROOT), '--', 'bash', '-c',
                       'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                       'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; '
                       'export PYTHONPATH="$1"; shift; '
                       'exec taskset -c 112-127 /opt/gfx-deps/venv/bin/python "$@"',
                       'bash', python_path, container_path(driver), component, '--label', label,
                       '--grant', container_path(grant_path), '--output', container_path(run / f'{component}-{label}.json')]
            prefix = run / f'{component}-{label}'
            prefix.with_suffix('.command').write_text(shlex.join(command) + '\n')
            with prefix.with_suffix('.log').open('x') as output:
                worker = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
                prefix.with_suffix('.pid').write_text(str(worker.pid) + '\n')
                actual_exit = worker.wait()
            prefix.with_suffix('.exit').write_text(str(actual_exit) + '\n')
            print(f'{component}/{label}: actual exit {actual_exit}', flush=True)
            if actual_exit:
                raise SystemExit(actual_exit)
    (run / 'execution-complete.json').write_text(json.dumps({'actual_exit': 0, 'GPU': 11,
                                                           'independent_bitwise_audit_pending': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()

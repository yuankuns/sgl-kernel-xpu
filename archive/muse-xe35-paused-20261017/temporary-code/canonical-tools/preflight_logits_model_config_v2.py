import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    driver = ROOT / 'benchmark_logits_model_config_v2.py'
    ast.parse(driver.read_text())
    command = [sys.executable, '-S', str(driver), '--label', 'before', '--phase', 'accuracy',
               '--grant', str(ROOT / 'nonexistent-model-config-grant-v2.json'),
               '--output', str(ROOT / 'nonexistent-model-config-output-v2.json')]
    aliases = ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
               'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS')
    tests = [({'ZE_AFFINITY_MASK': '0'}, 'Wrong main allocation'),
             ({'ZE_AFFINITY_MASK': '8'}, 'Wrong main allocation')]
    tests.extend(({'ZE_AFFINITY_MASK': '11', alias: '16'}, 'B8-only') for alias in aliases)
    tests.extend(({'ZE_AFFINITY_MASK': '11', key: value}, 'Wrong production scope')
                 for key, value in {'ISL': '2048', 'OSL': '32', 'TP': '2', 'EP': '2'}.items())
    results = []
    for overrides, expected in tests:
        environment = dict(os.environ)
        for key in (*aliases, 'ZE_AFFINITY_MASK', 'ISL', 'OSL', 'TP', 'EP'):
            environment.pop(key, None)
        environment.update(overrides)
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        process = subprocess.run(command, env=environment, capture_output=True, text=True)
        assert process.returncode != 0 and expected in process.stderr
        assert 'No module named' not in process.stderr
        results.append({'overrides': overrides, 'actual_exit': process.returncode,
                        'stderr': process.stderr, 'rejected_before_torch_import': True})
    report = {'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
              'ast_pass': True, 'negative_guards': results, 'GPU_started': False,
              'precision_threshold_changed': False, 'verified': True}
    destination = ROOT / 'logits-model-config-v2-cpu-preflight.json'
    with destination.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verified': True, 'negative_guards': len(results)}))


if __name__ == '__main__':
    main()

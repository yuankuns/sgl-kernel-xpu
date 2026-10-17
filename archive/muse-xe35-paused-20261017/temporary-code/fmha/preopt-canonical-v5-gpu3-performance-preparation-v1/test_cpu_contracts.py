import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    tests = [(['--guard-only'], {}, 0), (['--run'], {}, 2), (['--guard-only', '--gpu-id', '10'], {}, 2),
             (['--guard-only', '--limit', '1'], {}, 2), (['--guard-only'], {'BATCHES': '1'}, 1),
             (['--guard-only'], {'OSL': '32'}, 1), (['--guard-only'], {'DTYPE': 'FP16'}, 1)]
    rows = []
    for index, (arguments, overrides, expected) in enumerate(tests):
        child = subprocess.run([sys.executable, '-S', str(ROOT / 'prepare_cpu.py'), *arguments], env=dict(os.environ, **overrides), capture_output=True, text=True)
        assert child.returncode == expected
        assert 'torch/' not in child.stderr
        (ROOT / ('negative-contract-' + str(index) + '.log')).write_text(child.stdout + child.stderr)
        rows.append({'arguments': arguments, 'overrides': overrides, 'actual_exit': child.returncode, 'before_Torch_or_GPU': True})
    (ROOT / 'CPU_CONTRACTS.json').write_text(json.dumps({'actual_exit': 0, 'tests': rows, 'GPU_started': False}, indent=2) + '\n')
    print('7 CPU-only/no-runtime/no-subset/production guards0')


if __name__ == '__main__':
    main()

import os
import subprocess
import sys
from contracts import ROOT, write_json


def main():
    cases = [(['--guard-only'], {}, 0), (['--run'], {}, 1), (['--guard-only', '--serving-batch', '1'], {}, 2),
             (['--guard-only'], {'BATCHES': '1'}, 1), (['--guard-only'], {'OSL': '32'}, 1)]
    rows = []
    for index, (arguments, changes, expected) in enumerate(cases):
        environment = dict(os.environ, **changes)
        completed = subprocess.run([sys.executable, '-B', str(ROOT / 'accuracy_driver.py'), *arguments], env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (ROOT / ('CPU-contract-' + str(index) + '.log')).write_text(completed.stdout)
        assert completed.returncode == expected
        assert 'torch/' not in completed.stdout and 'sgl_kernel' not in completed.stdout
        rows.append({'arguments': arguments, 'overrides': changes, 'actual_exit': completed.returncode, 'before_Torch_or_GPU': True})
    write_json(ROOT / 'CPU_CONTRACT_TESTS.json', {'actual_exit': 0, 'tests': rows, 'GPU_started': False})
    print('CPU guard/no-grant/servingoverride contracts0; no GPU')


if __name__ == '__main__':
    main()

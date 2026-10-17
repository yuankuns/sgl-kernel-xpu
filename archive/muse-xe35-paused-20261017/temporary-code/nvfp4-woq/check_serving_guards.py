import json
import os
import pathlib
import subprocess
import sys

artifact = pathlib.Path(__file__).resolve().parent
target = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35')
script = target / 'benchmark/bench_nvfp4_woq_group_gemm.py'
base = [sys.executable, str(script), '--config', str(artifact / 'muse_config_xe35.json')]
cases = [(['--serving-batch', '7'], {}, 'invalid choice'), (['--tokens', '7'], {}, 'Muse requires B8')]
for name in ('BATCHES', 'BATCH', 'BATCH_SIZE', 'SERVING_BATCH', 'ISL', 'OSL', 'TP', 'EP'):
    cases.append(([], {name: '7'}, f'{name} must be'))
for arguments, overrides, expected in cases:
    environment = os.environ.copy()
    environment.update(overrides)
    result = subprocess.run([*base, *arguments], env=environment, text=True, capture_output=True)
    assert result.returncode != 0, (arguments, overrides, result.stdout)
    assert expected in result.stderr, (arguments, overrides, result.stderr)
    assert 'XPU' not in result.stderr and 'Level Zero' not in result.stderr, result.stderr
    print(json.dumps({'arguments': arguments, 'overrides': overrides, 'exit': result.returncode, 'expected_failure': expected, 'stderr': result.stderr}), flush=True)

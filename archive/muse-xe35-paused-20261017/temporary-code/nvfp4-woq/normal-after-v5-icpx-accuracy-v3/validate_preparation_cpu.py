import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import canonical_check_v5
import providers
import supervisor


DIRECTORY = Path(__file__).resolve().parent
assert 'torch' not in sys.modules
for path in DIRECTORY.glob('*.py'):
    ast.parse(path.read_text())
original = ast.parse((DIRECTORY.parent / 'source-native-icpx-v2-accuracy-v1/run_source_case.py').read_text())
current = ast.parse((DIRECTORY / 'run_accuracy_case.py').read_text())
for name in ('full_tensor_evidence', 'tensor_hash_xe35'):
    baseline = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
    updated = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert ast.dump(baseline, include_attributes=False) == ast.dump(updated, include_attributes=False)
shape = lambda tree: next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'shapes' for target in node.targets))
assert ast.dump(shape(original), include_attributes=False) == ast.dump(shape(current), include_attributes=False)
original_dtypes = sorted(ast.dump(node, include_attributes=False) for node in ast.walk(original) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr in ('bfloat16', 'float8_e4m3fn', 'float32', 'uint8', 'int8', 'int32'))
current_dtypes = sorted(ast.dump(node, include_attributes=False) for node in ast.walk(current) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr in ('bfloat16', 'float8_e4m3fn', 'float32', 'uint8', 'int8', 'int32'))
assert original_dtypes == current_dtypes
previous = os.environ.get('BATCHES')
os.environ['BATCHES'] = '7'
try:
    providers.preflight()
    raise AssertionError('Invalid serving override accepted')
except RuntimeError:
    assert 'torch' not in sys.modules
finally:
    if previous is None:
        del os.environ['BATCHES']
    else:
        os.environ['BATCHES'] = previous
try:
    canonical_check_v5.select_version('canonical-v4-icpx')
    raise AssertionError('Old v4 accepted')
except RuntimeError:
    pass
canonical_check_v5.select_version('canonical-v5-icpx')
frozen = canonical_check_v5.freeze('canonical-v5-icpx')
assert frozen['source_digest'] == supervisor.SOURCE_DIGEST
assert canonical_check_v5.VERSION_POOLS == {'canonical-v5-icpx': 'canonical-icpx-pool-v4'}
assert providers.source_result()['READY_for_fresh_SOURCE_accuracy_baseline']
previous = os.environ.pop('NV_V5_ACCURACY_GRANT', None)
try:
    providers.runtime_grant()
    raise AssertionError('Missing grant accepted')
except RuntimeError:
    assert 'torch' not in sys.modules
finally:
    if previous is not None:
        os.environ['NV_V5_ACCURACY_GRANT'] = previous
command = supervisor.wrapper('--suite', Path('/home/syk/test-grant.json'), gpu=True)
assert command[command.index('--gpu') + 1] == '9' and '--gpu-lock' in command
assert command[command.index('--workdir') + 1] == '/root'
assert command[-4:] == ['--mode', 'suite', '--grant', '/root/test-grant.json']
assert '/workspace' not in ' '.join(command)
for name in ('providers.py', 'accuracy_plugin.py', 'run_accuracy_case.py', 'supervisor.py'):
    text = (DIRECTORY / name).read_text()
    assert all(token not in text for token in ('torch.xpu.Event', 'device_samples', '--warmup', '--iterations', 'CUDAGraph', 'load_library'))
    assert 'performance_allowed' not in text or "'performance_allowed': False" in text
result = subprocess.run([sys.executable, '-B', str(DIRECTORY / 'supervisor.py'), '--version', 'canonical-v4-icpx', '--mode', 'launch'], capture_output=True, text=True)
assert result.returncode == 2
assert not (DIRECTORY / 'GPU-wrapper.started.json').exists()
assert not (DIRECTORY / 'CPU-proof.json').exists()
assert not (DIRECTORY / 'grant.json').exists()
assert 'torch' not in sys.modules
digests = {name: providers.digest(DIRECTORY / name) for name in supervisor.RECIPES}
print(json.dumps({'actual_preparation_CPU_check_exit': 0, 'version': 'canonical-v5-icpx', 'real_pool': 'canonical-icpx-pool-v4', 'source_digest': frozen['source_digest'], 'SOURCE_READY_sha256': providers.SOURCE_READY_SHA, 'scripts_sha256': digests, 'AST_verified': True, 'same_reviewed_tail_and_production_shape_AST': True, 'dtype_AST_unchanged': True, 'strict_uint8_bitwise_and_input_hash_functions_unchanged': True, 'old_v4_rejected': True, 'non_B8_rejected_before_Torch': True, 'no_grant_rejected_before_GPU': True, 'wrapper_GPU9_root_paths_checked': True, 'actual_normal_v5_package_import_verified': False, 'CPU_READY_for_runtime': False, 'main_grant_received': False, 'GPU_started': False, 'compiler_started': False, 'Torch_imported': False}, indent=2))

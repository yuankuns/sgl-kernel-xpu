import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from release import execute_frozen, FOLDER, FROZEN, maintenance_gate


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(16)):
        raise RuntimeError('BF CPU0–15 protocol required')
    original = {'__name__': 'CPU_original_BF_interface', '__file__': str(FROZEN / 'checks.py')}
    execute_frozen('checks.py', original)
    result = original['grant_preflight'](FOLDER.parent, FROZEN / 'fresh-grant.json', False)
    assert isinstance(result, tuple) and len(result) == 4
    manifest, manifest_path, site, envelope = result
    assert envelope['grant'] == json.loads((FROZEN / 'fresh-grant.json').read_text())
    assert original['digest'](manifest_path) == '973f593bdc4603895a0708fa953b70ce8184e7f7cef126c717ed107078c84414'
    tree = ast.parse((FOLDER / 'checks.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'grant_preflight')
    assert function.body[0].targets[0].id == 'result' and function.body[-1].value.id == 'result'
    namespace = {'__name__': 'CPU_original_BF_method', '__file__': str(FOLDER / 'run_case.py')}
    execute_frozen('run_case.py', namespace)
    previous = sys.argv[:]
    previous_mask = os.environ.get('ZE_AFFINITY_MASK')
    os.environ['ZE_AFFINITY_MASK'] = '4'
    try:
        sys.argv = ['CPU_protocol', 'source', 'canonical-v4-icpx-CPU-cleanconfirm', '--grant-file', str(FOLDER / 'fresh-grant.json')]
        args = namespace['parse_args_xe35']()
        assert (args.calls, args.repeat, args.warmup, args.serving_batch, args.isl, args.osl, args.tp, args.ep) == (20, 20, 5, 8, 4096, 16, 1, 1)
    finally:
        sys.argv = previous
        if previous_mask is None:
            os.environ.pop('ZE_AFFINITY_MASK', None)
        else:
            os.environ['ZE_AFFINITY_MASK'] = previous_mask
    body = (FROZEN / 'run_case.py').read_text()
    assert 'torch.Generator().manual_seed(42)' in body
    assert 'for columns in [8192, 18432, 202048]:' in body
    assert "names = ['native_kernel', 'packed_kernel', 'native_api', 'packed_api']" in body
    assert 'atol=5e-3, rtol=5e-3' in body
    import torch
    torch.set_num_threads(8)
    hashes = []
    for role in ['source', 'port']:
        generator = torch.Generator().manual_seed(42)
        rows = []
        for columns in [8192, 18432, 202048]:
            activation = torch.randn((8, 8192), dtype=torch.bfloat16, generator=generator)
            weight = torch.randn((columns, 8192), dtype=torch.bfloat16, generator=generator)
            rows.append({'N': columns, 'activation_sha256': hashlib.sha256(memoryview(activation.view(torch.uint8).numpy())).hexdigest(), 'weight_sha256': hashlib.sha256(memoryview(weight.view(torch.uint8).numpy())).hexdigest(), 'activation_elements': activation.numel(), 'weight_elements': weight.numel()})
            del activation, weight
        hashes.append(rows)
    assert hashes[0] == hashes[1]
    assert not torch.xpu._initialized
    record = {'epoch': time.time(), 'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'real_original_interface_four_tuple': True, 'fresh_wrapper_preserves_return_contract': True, 'old_grant_CPU_contract_test_only_not_GPU_consumption': True, 'protocol': 'original reviewed one complete BAAB/all12 paths/20calls/20samples/warm5', 'full_CPU_input_bytes_for_all3_shapes_equal': True, 'input_hashes': hashes[0], 'input_provenance': 'Original frozen seed42 BF16 generation sequentially recomputed SOURCE/PORT on CPU; not claimed historical saved input files', 'GPU_started': False, 'maintenance': maintenance_gate()}
    (FOLDER / 'CPU_PROTOCOL_INPUT_INTERFACE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'CPU_protocol_interface_inputs_exit': 0, 'full_input_byte_comparison': True, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()

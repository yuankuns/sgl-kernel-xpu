import ast
import hashlib
import json
import os
from pathlib import Path
import runpy


def main():
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    directory = Path(__file__).resolve().parent
    site = directory.parent / 'fresh-current-source-icpx-native-v3/site'
    assert Path(sgl_kernel.__file__).resolve() == site / 'sgl_kernel/__init__.py'
    assert Path(torch.__file__).resolve().is_relative_to('/opt/gfx-deps/venv')
    assert not torch.xpu.is_initialized() and torch.get_default_device().type == 'cpu'
    assert torch.get_default_dtype() == torch.float32 and sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    torch.set_num_threads(8)
    source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests/test_per_token_group_quant_mxfp4.py')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == '4656ddb0d3161582c5d0a2015cdf8bd5b2b563e8653e929b7286b7b1e1cd1aab'
    original = runpy.run_path(str(source))
    quantize = original['quantize_to_mxfp4']
    functions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                 for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.FunctionDef)
                 and node.name in ['test_fused_swiglu_matches_separate_dtype_surface', 'test_fused_swiglu_cpu_reference']}
    reports = []
    for kind, hidden in [('surface', 32), ('reference', 256), ('reference', 8192)]:
        if kind == 'surface':
            values = torch.zeros(1, 64, dtype=torch.bfloat16, device='cpu')
            values[0, 0], values[0, 32] = 0.5, 2.40625
            values[0, 1], values[0, 33] = 4.0, 1.25
            node = 'test_fused_swiglu_matches_separate_dtype_surface[dtype0-0.5-2.40625]'
            seed = None
        else:
            seed = 20261016 + hidden
            generator = torch.Generator().manual_seed(seed)
            values = torch.randn(8, 2 * hidden, generator=generator).to(torch.bfloat16)
            node = f'test_fused_swiglu_cpu_reference[{hidden}-dtype0]'
        fp32_activation = torch.nn.functional.silu(values[:, :hidden].float()) * values[:, hidden:].float()
        activation = fp32_activation.to(torch.bfloat16)
        expected_q, expected_s = quantize(activation)
        tensors = {'original_input': values, 'original_CPU_FP32_activation': fp32_activation,
                   'original_CPU_BF16_activation': activation, 'expected_q': expected_q, 'expected_s': expected_s}
        filename = f'CPU-{kind}-{hidden}.pt'
        torch.save(tensors, directory / filename)
        reports.append({'kind': kind, 'hidden': hidden, 'node_suffix': node, 'seed': seed, 'fixture_file': filename,
                        'tensors': {name: tensor_sha256(value) for name, value in tensors.items()},
                        'source_sha256': source_sha, 'reference': 'original quantize_to_mxfp4, unchanged defaults',
                        'precision_gate': {'atol': 0, 'rtol': 0}, 'serving_scope': 'B8/ISL4096/OSL16/TP1/EP1; surface1 is original generic tensor edge, not serving B1'})
    assert not torch.xpu.is_initialized()
    report = {'fixtures': reports, 'original_test_function_AST_sha256': functions, 'SOURCE_tests_GPU_executed': False,
              'GPU_initialized': False, 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__,
              'package': sgl_kernel.__file__, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'no_monkeypatch': True}
    (directory / 'CPU-original-fixtures.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

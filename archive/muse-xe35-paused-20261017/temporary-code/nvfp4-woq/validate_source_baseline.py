import importlib.util
import json
import pathlib
import time

import torch

from compare_source_port import serving_scope_xe35

serving_scope_xe35()
import source_nvfp4_woq_xe35 as source

root = pathlib.Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
spec = importlib.util.spec_from_file_location('original_nvfp4_reference_xe35', root / 'tests/nvfp4_utils.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(8)
print(json.dumps({'source_bridge': source.__file__, 'gpu': torch.xpu.get_device_name(0), 'source': 'd983e87', 'scope': 'B8/ISL4096/OSL16/TP1/EP1', 'cases': 'generic kernel tails, not serving overrides'}), flush=True)
for counts, channels, inner in (([1, 7, 0, 3], 192, 64), ([33, 31], 256, 128)):
    torch.xpu.synchronize()
    time.sleep(2)
    torch.manual_seed(42)
    experts = len(counts)
    packed = torch.randint(0, 256, (experts, channels, inner // 2), dtype=torch.uint8)
    scales_cpu = torch.randint(1, 16, (experts, channels, inner // 16)).float().to(torch.float8_e4m3fn)
    globals_cpu = torch.arange(1, experts + 1, dtype=torch.float32).reciprocal()
    expected_c, expected_sfmx = reference.nvfp4_woq_prep_ref(scales_cpu, globals_cpu)
    scales = scales_cpu.to('xpu')
    globals_dq = globals_cpu.to('xpu')
    multipliers = torch.empty_like(expected_c, device='xpu')
    exponents = torch.empty_like(expected_sfmx, device='xpu')
    source.prep_xe35(multipliers, exponents, scales, globals_dq)
    torch.xpu.synchronize()
    assert torch.equal(multipliers.cpu(), expected_c)
    assert torch.equal(exponents.cpu(), expected_sfmx)
    weights = packed.to('xpu')
    problems = torch.tensor([[count, channels, inner] for count in counts], dtype=torch.int32, device='xpu')
    offsets = torch.tensor([sum(counts[:expert]) for expert in range(experts)], dtype=torch.int32, device='xpu')
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='xpu')
    for mode in ('bf16', 'mxfp8'):
        torch.xpu.synchronize()
        time.sleep(2)
        activations_cpu = torch.randint(-4, 5, (sum(counts), inner)).float()
        activations_cpu = activations_cpu.to(torch.bfloat16 if mode == 'bf16' else torch.float8_e4m3fn)
        activations = activations_cpu.to('xpu')
        output = torch.empty((sum(counts), channels), dtype=torch.float32, device='xpu')
        if mode == 'bf16':
            source.bf16_xe35(output, activations, weights, exponents, multipliers, problems, offsets, workspace)
            decoded = reference.dequantize_nvfp4_via_bf16_prepared(packed, expected_c, expected_sfmx)
        else:
            activation_scales = torch.full((sum(counts), inner // 32), 127, dtype=torch.uint8, device='xpu')
            source.mxfp8_xe35(output, activations, weights, activation_scales, exponents, multipliers, problems, offsets, workspace)
            requantized, encoded = reference.nvfp4_to_mxfp8_ref(packed, scales_cpu, globals_cpu)
            decoded = reference.dequantize_mxfp8(requantized, encoded)
        torch.xpu.synchronize()
        expected = torch.empty_like(output, device='cpu')
        start = 0
        for expert, count in enumerate(counts):
            expected[start:start + count] = activations_cpu[start:start + count].float() @ decoded[expert].float().T
            start += count
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        print(json.dumps({'mode': mode, 'counts': counts, 'n': channels, 'k': inner, 'full_output': True, 'full_scales': True, 'max_abs': (actual - expected).abs().max().item(), 'rtol': 5e-3, 'atol': 5e-3}), flush=True)

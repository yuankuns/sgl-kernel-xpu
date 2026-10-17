import json
import time

import torch

from compare_source_port import device_samples_xe35, serving_scope_xe35

serving_scope_xe35()
import source_nvfp4_woq_xe35 as source

counts = [1, 7, 3]
experts, channels, inner = len(counts), 256, 128
torch.manual_seed(42)
weights = torch.randint(0, 256, (experts, channels, inner // 2), device='xpu', dtype=torch.uint8)
scales = torch.ones((experts, channels, inner // 16), device='xpu').to(torch.float8_e4m3fn)
globals_dq = torch.ones(experts, device='xpu')
multipliers = torch.empty_like(scales, dtype=torch.bfloat16)
exponents = torch.empty((experts, inner // 32, channels), device='xpu', dtype=torch.uint8)
source.prep_xe35(multipliers, exponents, scales, globals_dq)
problems = torch.tensor([[count, channels, inner] for count in counts], device='xpu', dtype=torch.int32)
offsets = torch.tensor([sum(counts[:expert]) for expert in range(experts)], device='xpu', dtype=torch.int32)
workspace = torch.empty(64 * 1024 * 1024, device='xpu', dtype=torch.uint8)
for mode in ('bf16', 'mxfp8'):
    torch.xpu.synchronize()
    time.sleep(2)
    dtype = torch.bfloat16 if mode == 'bf16' else torch.float8_e4m3fn
    activations = torch.randint(-4, 5, (sum(counts), inner), device='xpu', dtype=torch.int8).to(dtype)
    output = torch.empty((sum(counts), channels), device='xpu', dtype=torch.float32)
    if mode == 'bf16':
        function = lambda: source.bf16_xe35(output, activations, weights, exponents, multipliers, problems, offsets, workspace)
    else:
        activation_scales = torch.full((sum(counts), inner // 32), 127, device='xpu', dtype=torch.uint8)
        function = lambda: source.mxfp8_xe35(output, activations, weights, activation_scales, exponents, multipliers, problems, offsets, workspace)
    samples = device_samples_xe35(function, 2, 5)
    assert all(value > 0 for value in samples), samples
    print(json.dumps({'mode': mode, 'timer': 'original XPU event structure', 'samples_us': samples, 'generic_shape': [experts, sum(counts), channels, inner], 'gpu_mask': '5', 'not_production_performance': True}), flush=True)

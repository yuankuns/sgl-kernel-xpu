import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from input_forensics import tensor_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    cache = directory / 'production-cpu-cache-v2'
    cache.mkdir()
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80)) and not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests/test_per_token_group_quant_mxfp4.py')
    reference = runpy.run_path(str(source))['quantize_to_mxfp4']

    def quant_all(values):
        codes = torch.empty(values.shape[0], 2048, dtype=torch.uint8)
        scales = torch.empty(values.shape[0], 128, dtype=torch.uint8)
        for start in range(0, values.shape[0], 256):
            current_q, current_s = reference(values[start:start + 256])
            codes[start:start + 256] = current_q
            scales[start:start + 256] = current_s
        return codes, scales

    records = []
    for tokens in [8, 32768]:
        generator = torch.Generator().manual_seed(20261017 + tokens)
        logits = torch.randn(tokens, 256, dtype=torch.float32, generator=generator)
        bias = torch.randn(256, dtype=torch.float32, generator=generator)
        scores = torch.sigmoid(logits * 0.3535533905932738)
        ids = torch.topk(scores + bias, 8, dim=-1, sorted=True).indices.int()
        selected = scores.gather(1, ids.long())
        weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-15)
        route = {'logits': logits, 'bias': bias, 'CPU_reference_ids': ids, 'CPU_reference_weights': weights}
        route_path = cache / f'route-{tokens}.pt'
        torch.save(route, route_path)
        records.append({'shape': f'route-{tokens}-256-8-fp32', 'file': str(route_path), 'file_SHA256': digest(route_path),
                        'tensors': {name: tensor_sha256(value) for name, value in route.items()},
                        'CPU_route_reference_scope': 'new full CPU scores/ranking/weights; original runtime device-ranking assertion unchanged; no legacy golden'})
        del route, logits, bias, ids, weights, scores, selected
        hidden = torch.randn(tokens, 4096, generator=generator).to(torch.bfloat16)
        codes, scales = quant_all(hidden)
        gather_path = cache / f'gather-{tokens}.pt'
        gather = {'hidden': hidden, 'CPU_reference_q_by_input_row': codes, 'CPU_reference_s_by_input_row': scales}
        torch.save(gather, gather_path)
        rows = tokens * 8 + 3 * min(256, tokens * 8)
        records.append({'shape': f'gather-quant-map-{rows}-4096-bf16', 'file': str(gather_path), 'file_SHA256': digest(gather_path),
                        'tensors': {name: tensor_sha256(value) for name, value in gather.items()},
                        'full_reference_scope': 'all original input rows/full4096 columns; original runtime actual route-derived map selects these refs, scale transpose/padding remain original'})
        del gather, hidden, codes, scales
        gate_up = torch.randn(rows, 8192, generator=generator).to(torch.bfloat16)
        activation = torch.empty(rows, 4096, dtype=torch.bfloat16)
        for start in range(0, rows, 256):
            block = gate_up[start:start + 256]
            activation[start:start + 256] = (torch.nn.functional.silu(block[:, :4096].float()) * block[:, 4096:].float()).to(torch.bfloat16)
        codes, scales = quant_all(activation)
        swiglu_path = cache / f'swiglu-{tokens}.pt'
        swiglu = {'gate_up': gate_up, 'CPU_reference_activation': activation,
                  'CPU_reference_q_by_input_row': codes, 'CPU_reference_s_by_input_row': scales}
        torch.save(swiglu, swiglu_path)
        records.append({'shape': f'swiglu-quant-map-{rows}-4096-bf16', 'file': str(swiglu_path), 'file_SHA256': digest(swiglu_path),
                        'tensors': {name: tensor_sha256(value) for name, value in swiglu.items()},
                        'full_reference_scope': 'all allocated original rows/full4096 activation columns and group32 original CPU quant references; runtime compares valid q/used physical scales only'})
        del swiglu, gate_up, activation, codes, scales
        print(json.dumps({'tokens': tokens, 'new_CPU_input_and_full_reference_cache_complete': True}), flush=True)
    assert len(records) == 6 and not torch.xpu.is_initialized()
    manifest = {'actual_CPU_cache_exit': 0, 'production_shapes': 6, 'records': records,
                'original_reference_source_SHA256': digest(source), 'original_group_size': 32,
                'row_chunking_scope': 'group32 reference has no cross-row reduction; original helper/default precision used for every row',
                'original_seed_and_rng_consumption_preserved': True, 'legacy_native_or_output_goldens_used': False,
                'GPU_initialized': False, 'no_compile_or_GPU_work': True}
    (directory / 'CPU-production-cache-v2.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

import json
import os
from pathlib import Path
import runpy
import statistics
import time

from check_official import canonical_preflight
from check_preopt import BASE, REQUIRED, preflight, verify_loaded


ART = Path(__file__).resolve().parent
ROOT = ART.parent.parent / 'pr/muse-fmha-xe35'


def cpu_reference(case, tensors, args, torch):
    query, key, value, table, _, _, output = tensors
    query = query.cpu().float()
    key = key.cpu().float()
    value = value.cpu().float()
    table = table.cpu().long()
    output = output.cpu().float()
    positions = torch.arange(case.cache_length, device='cpu')
    checked, max_abs = 0, 0.0
    started = time.monotonic()
    for request in range(case.requests):
        keys = key[table[request]].reshape(-1, 16, 64)[:case.cache_length].transpose(0, 1)
        values = value[table[request]].reshape(-1, 16, 64)[:case.cache_length].transpose(0, 1)
        for offset in range(0, case.query_length, args.chunk):
            indices = torch.arange(offset, min(offset + args.chunk, case.query_length), device='cpu')
            queries = query[request * case.query_length + indices].reshape(-1, 16, 8, 64).permute(1, 2, 0, 3)
            scores = torch.matmul(queries, keys[:, None].transpose(-1, -2)) * (64 ** -0.5)
            absolute = indices + case.cache_length - case.query_length
            mask = positions[None, :] > absolute[:, None]
            if case.window >= 0:
                mask |= positions[None, :] < absolute[:, None] - case.window
            probabilities = scores.masked_fill(mask, -float('inf')).softmax(dim=-1)
            expected = torch.matmul(probabilities, values[:, None]).permute(2, 0, 1, 3).reshape(-1, 128, 64)
            observed = output[request * case.query_length + indices]
            assert torch.isfinite(expected).all() and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, expected, atol=5e-3, rtol=5e-3)
            checked += observed.numel()
            max_abs = max(max_abs, (observed - expected).abs().max().item())
    assert checked == case.requests * case.query_length * 128 * 64
    return {'reference': 'independent CPU FP32 masked matmul/softmax/matmul on actual BF16 inputs',
            'reference_device': 'cpu', 'coverage': 'full', 'query_positions': 'all',
            'checked_elements': checked, 'max_abs': max_abs, 'atol': 5e-3, 'rtol': 5e-3,
            'cpu_reference_wall_seconds': time.monotonic() - started}


def main():
    bench = runpy.run_path(str(ROOT / 'benchmark/bench_muse_fmha.py'))
    args, cases = bench['arguments_xe35']()
    if args.guard_only:
        return
    assert args.reference == 'full' and not args.paired_dir
    provider = os.environ['FMHA_COMPARISON_PROVIDER']
    manifest, manifest_path, canonical_package, required = canonical_preflight()
    if provider == 'preopt':
        package, hashes = preflight()
        required = REQUIRED
    else:
        assert provider == 'port'
        package = canonical_package
        hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
    assert os.environ.get('ZE_AFFINITY_MASK') == '5'
    import torch
    import sgl_kernel
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.utils import is_xe3_arch

    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    verify_loaded(package, hashes, required)
    assert bench['is_fmha_arch_xe35'](torch, is_xe3_arch), 'dedicated acceptance requires Xe35'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    result = {'label': args.label, 'provider': provider,
              'preopt_base_sha': BASE, 'canonical_source_digest': manifest['source']['source_digest'],
              'canonical_manifest': str(manifest_path), 'workload': 'B8 ISL4096 OSL16 TP1 EP1 BF16',
              'timing': 'device events from frozen benchmark; not CPU reference wall-time', 'cases': []}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    with torch.inference_mode():
        for case in cases:
            print(json.dumps({'starting_case': case.name, 'provider': provider, 'mode': args.mode}), flush=True)
            torch.xpu.synchronize()
            time.sleep(2)
            tensors = bench['tensors_xe35'](case, torch)
            launch = lambda: bench['launch_xe35'](case, tensors, flash_attn_with_kvcache)
            record = {'name': case.name, 'requests': case.requests, 'query_length': case.query_length,
                      'cache_length': case.cache_length, 'window': case.window}
            if args.mode == 'accuracy':
                launch()
                torch.xpu.synchronize()
                record.update(cpu_reference(case, tensors, args, torch))
                record['accuracy'] = 'PASS'
            else:
                samples = bench['measure_xe35'](launch, args, torch)
                record.update(samples_ms=samples, median_ms=statistics.median(samples), warmup=args.warmup,
                              iterations=args.iterations, graph_replays=args.replays)
            result['cases'].append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            torch.xpu.synchronize()
            del tensors, launch
    verify_loaded(package, hashes, required)
    result['provenance'] = bench['provenance_xe35'](sgl_kernel, torch)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

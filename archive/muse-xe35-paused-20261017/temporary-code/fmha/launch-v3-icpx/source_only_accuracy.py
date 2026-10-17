import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

from baseline_compiler_provenance import ARCHIVE, baseline_provenance
from check_preopt import REQUIRED, verify_loaded
from matched_baseline_provenance import package_preflight, provenance


ART = Path(__file__).resolve().parent.parent
PR = ART.parent.parent / 'pr/muse-fmha-xe35'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def provider_preflight(provider):
    if provider == 'historical-optimized':
        assert os.environ.get('FMHA_BASELINE_KIND', 'archived') == 'archived'
        compiler = baseline_provenance('source')
        package = ARCHIVE / 'site-after/sgl_kernel'
        records = json.loads((ARCHIVE / 'binary-manifest-after.json').read_text())
        hashes = {}
        for record in records:
            name = Path(record['path']).name
            assert digest(package / name) == record['sha256'], name
            hashes[name] = record['sha256']
        assert len(hashes) == 216
        return package, hashes, compiler
    role = 'source' if provider == 'fresh-optimized' else 'preopt'
    assert provider in ('fresh-optimized', 'fresh-preopt')
    package, hashes = package_preflight(role)
    return package, hashes, provenance(role)


def tensor_digest(tensor, torch):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def full_cpu_reference(case, tensors, torch):
    query, key, value, table, _, _, output = tensors
    query, key, value = query.cpu().float(), key.cpu().float(), value.cpu().float()
    table, output = table.cpu().long(), output.cpu().float()
    positions = torch.arange(case.cache_length, device='cpu')
    checked, max_abs, failed_chunks, first_failure = 0, 0.0, 0, None
    for request in range(case.requests):
        keys = key[table[request]].reshape(-1, 16, 64)[:case.cache_length].transpose(0, 1)
        values = value[table[request]].reshape(-1, 16, 64)[:case.cache_length].transpose(0, 1)
        for offset in range(0, case.query_length, 64):
            indices = torch.arange(offset, min(offset + 64, case.query_length), device='cpu')
            queries = query[request * case.query_length + indices].reshape(-1, 16, 8, 64).permute(1, 2, 0, 3)
            scores = torch.matmul(queries, keys[:, None].transpose(-1, -2)) * (64 ** -0.5)
            absolute = indices + case.cache_length - case.query_length
            mask = positions[None, :] > absolute[:, None]
            if case.window >= 0:
                mask |= positions[None, :] < absolute[:, None] - case.window
            probabilities = scores.masked_fill(mask, -float('inf')).softmax(dim=-1)
            expected = torch.matmul(probabilities, values[:, None]).permute(2, 0, 1, 3).reshape(-1, 128, 64)
            observed = output[request * case.query_length + indices]
            try:
                assert torch.isfinite(expected).all() and torch.isfinite(observed).all()
                torch.testing.assert_close(observed, expected, atol=5e-3, rtol=5e-3)
            except AssertionError as error:
                failed_chunks += 1
                if first_failure is None:
                    first_failure = {'request': request, 'query_chunk_start': offset, 'error': repr(error)}
            checked += observed.numel()
            max_abs = max(max_abs, (observed - expected).abs().max().item())
    assert checked == case.requests * case.query_length * 128 * 64
    return {'reference': 'independent CPU FP32 masked matmul/softmax/matmul on original BF16 input',
            'coverage': 'full', 'query_positions': 'all', 'heads_channels_requests': 'all',
            'checked_elements': checked, 'max_abs': max_abs, 'atol': 5e-3, 'rtol': 5e-3,
            'reference_PASS': failed_chunks == 0, 'failed_chunks': failed_chunks, 'first_failure': first_failure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=('historical-optimized', 'fresh-optimized', 'fresh-preopt'), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--cpu-import-only', action='store_true')
    args = parser.parse_args()
    bench = runpy.run_path(str(PR / 'benchmark/bench_muse_fmha.py'))
    _, cases = bench['arguments_xe35'](['--guard-only', '--mode', 'accuracy', '--reference', 'full'])
    assert len(cases) == 38 and len([case for case in cases if case.name.startswith('prefill')]) == 6
    if args.guard_only:
        print('B8 fixed scope, 38 unsliced cases, accuracy only; no timing options accepted')
        return
    package, hashes, compiler = provider_preflight(args.provider)
    if not args.cpu_import_only:
        grant = json.loads(args.grant.read_text())
        assert grant['phase'] == 'source-only-accuracy' and grant['gpu'] == 5 and grant['coordinator_release'] is True
        assert grant['canonical_version'] == 'canonical-v4-icpx'
        freeze = json.loads((ART / 'canonical-source-freeze-v4-icpx.json').read_text())
        assert grant['source_digest'] == freeze['source_digest']
        assert grant['epoch'] > grant['issued_epoch']
        assert grant['providers'] == ['historical-optimized', 'fresh-optimized', 'fresh-preopt']
        assert grant['workload'] == {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}
        assert grant['no_perf_warmup_events_graphs_PORT'] is True
        assert digest(ART / 'matched-native-icpx-v1/source/normal-package-manifest.json') == grant['fresh_source_manifest_sha256']
        assert digest(ART / 'matched-native-icpx-v1/preopt/normal-package-manifest.json') == grant['fresh_preopt_manifest_sha256']
        assert os.environ.get('ZE_AFFINITY_MASK') == '5'
        while time.time() < grant['epoch']:
            time.sleep(0.25)
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    verify_loaded(package, hashes, REQUIRED)
    if args.cpu_import_only:
        assert not torch.xpu.is_initialized()
        assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
        args.output.write_text(json.dumps({'provider': args.provider, 'CPU_import': True, 'gpu_initialized': False,
            'compiler': compiler, 'actual_package': sgl_kernel.__file__, 'required_DSOs': {name: hashes[name] for name in REQUIRED}}, indent=2) + '\n')
        print('normal CPU provider import PASS; XPU initialized=False')
        return
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.utils import is_xe3_arch
    assert bench['is_fmha_arch_xe35'](torch, is_xe3_arch), 'actual device must be Xe3.5; no fake architecture'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    result = {'provider': args.provider, 'label': args.label, 'grant_sha256': digest(args.grant),
              'scope': 'SOURCE-only accuracy; no PORT/performance/warmup/events/graphs',
              'workload': grant['workload'], 'compiler_provenance': compiler, 'cases': []}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    with torch.inference_mode():
        for case in cases:
            torch.xpu.synchronize()
            time.sleep(2)
            print(json.dumps({'starting_case': case.name, 'provider': args.provider}), flush=True)
            tensors = bench['tensors_xe35'](case, torch)
            bench['launch_xe35'](case, tensors, flash_attn_with_kvcache)
            torch.xpu.synchronize()
            actual = tensors[-1].cpu()
            record = {'name': case.name, 'requests': case.requests, 'query_length': case.query_length,
                      'cache_length': case.cache_length, 'window': case.window, 'native_launches': 1,
                      'input_sha256': [tensor_digest(tensor, torch) for tensor in tensors[:-1]],
                      'output_sha256': tensor_digest(actual, torch)}
            result['cases'].append(record)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            try:
                record.update(full_cpu_reference(case, tensors, torch))
                if args.provider == 'historical-optimized':
                    args.paired_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(actual, args.paired_dir / f'{case.name}.pt')
                    (args.paired_dir / f'{case.name}.json').write_text(json.dumps(record, indent=2) + '\n')
                elif args.provider == 'fresh-optimized':
                    original = torch.load(args.paired_dir / f'{case.name}.pt', map_location='cpu', weights_only=True)
                    original_record = json.loads((args.paired_dir / f'{case.name}.json').read_text())
                    assert record['input_sha256'] == original_record['input_sha256'], 'original input bytes differ'
                    assert actual.dtype == original.dtype == torch.bfloat16 and actual.shape == original.shape
                    assert torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)), 'full-output bitwise mismatch'
                    assert record['output_sha256'] == original_record['output_sha256']
                    record['full_output_bitwise'] = True
                record['accuracy'] = 'PASS' if record['reference_PASS'] else 'FAIL'
            except Exception as error:
                record['accuracy'] = 'FAIL'
                record['failure'] = repr(error)
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({'case_failure': case.name, 'error': repr(error)}), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            torch.xpu.synchronize()
            del tensors, actual
    verify_loaded(package, hashes, REQUIRED)
    result['actual_runtime_provenance'] = bench['provenance_xe35'](sgl_kernel, torch)
    result['all_38_reference_PASS'] = len(result['cases']) == 38 and all(record['accuracy'] == 'PASS' for record in result['cases'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_38_reference_PASS']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

import argparse
import ast
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'source-before-icpx-bounded-v1'
OUTPUT = ROOT / 'source-compiler-accuracy-v1'
MANIFEST = JOB / 'native-package-manifest.json'
BARRIER = ROOT / 'gpu7-source-compiler-accuracy-v1-barrier.json'
BENCH = ROOT.parent.parent / 'pr/muse-residual-norm-xe35/benchmark/bench_fused_norm_residual.py'
PHASE = 'residual-source-compiler-accuracy-v1'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def scope():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
                 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise ValueError(f'B8-only: {name} must be 8 before Torch/GPU')
    for name, expected in (('ISL', '4096'), ('OSL', '16'), ('TP', '1'), ('EP', '1')):
        if os.environ.get(name, expected) != expected:
            raise ValueError(f'{name} must be {expected} before Torch/GPU')


def audit():
    namespace = runpy.run_path(str(ROOT / 'check_canonical_package_v3.py'))
    record = namespace['before_provenance']()
    assert record['supervisor_actual_exit'] == record['native_build_actual_exit'] == record['normal_import_actual_exit'] == 0
    assert record['active_owned_build_processes'] == []
    assert record['flags_and_AOT_exact_match_historical'] is True
    original = json.loads((ROOT / 'source-compiler-provenance.json').read_text())
    site_record = json.loads((JOB / 'site-sha.json').read_text())
    old_site = ROOT / 'source-full-venv/lib/python3.12/site-packages/sgl_kernel'
    actual = {path.relative_to(old_site).as_posix(): digest(path)
              for path in old_site.rglob('*') if path.is_file()}
    assert actual == site_record['pre']
    assert actual['libsgl-ops-sycl-FusedNormResidual.so'] == original['source_native_sha256']
    return {'source_manifest_sha256': digest(MANIFEST), 'native_sha256': record['tested_native_sha256'],
            'historical_native_sha256': original['source_native_sha256'],
            'source_ref': record['source_ref'], 'original_cutlass_pin': record['cutlass']['pin'],
            'snapshot_digest': record['source_snapshot_digest'],
            'registration_DSO_sha256': record['registration_DSO_sha256'],
            'registration_GNU_not_rebuilt': True, 'site_manifest_sha256': digest(JOB / 'site-sha.json'),
            'oracle_benchmark_sha256': digest(BENCH), 'all_owned_compilers_drained': True,
            'official_AFTER': False, 'accuracy_only': True}


def grant():
    expected = json.loads((OUTPUT / 'cpu-preflight.json').read_text())
    ready = json.loads((OUTPUT / 'ready.json').read_text())
    released = json.loads(BARRIER.read_text())
    for name in ('source_manifest_sha256', 'native_sha256', 'historical_native_sha256'):
        assert released[name] == expected[name]
    assert digest(MANIFEST) == expected['source_manifest_sha256']
    assert released['gpu'] == 7 and released['coordinator_release'] is True
    assert released['accuracy_only'] is True and released['phase'] == PHASE
    assert released['scope'] == 'B8/ISL4096/OSL16/TP1/EP1'
    assert isinstance(released['start_epoch'], (float, int)) and not isinstance(released['start_epoch'], bool)
    assert ready['ready_epoch'] <= released['start_epoch'] < float('inf')
    return released


def normal_import(provider):
    assert os.environ['LD_LIBRARY_PATH'].split(':')[0] == '/opt/gfx-deps/profiling_tools/lib'
    import torch
    assert not torch.xpu.is_initialized()
    assert torch.ones(4, device='cpu').sum().item() == 4
    import sgl_kernel
    prefix = ROOT / 'source-full-venv' if provider == 'historical' else JOB / 'isolated-source-venv'
    site = prefix / 'lib/python3.12/site-packages/sgl_kernel'
    assert Path(sys.prefix).resolve() == prefix.resolve()
    assert Path(sgl_kernel.__file__).resolve() == (site / '__init__.py').resolve()
    expected = json.loads((JOB / 'site-sha.json').read_text())['pre' if provider == 'historical' else 'post']
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if '/sgl_kernel/' in line and ('.so' in line)}
    for path in mapped:
        assert digest(path) == expected[path.relative_to(site.resolve()).as_posix()]
    assert (site / 'libsgl-ops-sycl-FusedNormResidual.so').resolve() in mapped
    assert any(path.name.startswith('common_ops.') for path in mapped)
    op = torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm
    assert not torch.xpu.is_initialized()
    proof = {'provider': provider, 'prefix': str(prefix), 'package': str(sgl_kernel.__file__),
             'schema': str(op.default._schema), 'mapped_libraries': {str(path): digest(path) for path in sorted(mapped)},
             'GPU_initialized': False, 'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH'],
             'normal_original_SOURCE_registration': True, 'AFTER': False}
    return torch, sgl_kernel, op, proof


def tensor_digest(tensor):
    result = hashlib.sha256()
    for start in range(0, tensor.shape[0], 512):
        result.update(tensor[start:start + 512].contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def run(provider):
    released = grant()
    assert time.time() >= released['start_epoch']
    global torch
    torch, package, op, proof = normal_import(provider)
    assert os.environ.get('ZE_AFFINITY_MASK') == '7'
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if not torch.xpu.is_available():
        raise RuntimeError('XPU unavailable')
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch(), 'Xe35 compute capability 3.5 required'
    (OUTPUT / f'{provider}-normal-import.json').write_text(json.dumps(proof, indent=2) + '\n')
    tree = ast.parse(BENCH.read_text())
    names = {'round_bfloat16_xe35', 'reference_xe35', 'check_accuracy_xe35'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    oracle = {'torch': torch, 'hashlib': hashlib, 'POST_EPS': 1e-8, 'PRE_EPS': 1e-5}
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('POST_EPS', 'PRE_EPS')}
    assert constants == {name: oracle[name] for name in ('POST_EPS', 'PRE_EPS')}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BENCH), 'exec'), oracle)
    output_dir = OUTPUT / f'{provider}-outputs'
    output_dir.mkdir()
    results = []
    for rows, mode, rounded, wide in itertools.product([8, 32768], ['none', 'weight', 'input_scale'], [False, True], [False, True]):
        torch.manual_seed(42)
        branch = torch.randn(rows, 8192, device='xpu', dtype=torch.bfloat16)
        stream = torch.randn(rows, 8192, device='xpu', dtype=torch.float32)
        post = (torch.empty(0, device='xpu', dtype=torch.bfloat16) if mode == 'none'
                else torch.randn(8192, device='xpu', dtype=torch.bfloat16))
        pre = torch.randn(8192, device='xpu', dtype=torch.bfloat16)
        alpha = torch.rand(8192, device='xpu', dtype=torch.float32)
        beta = torch.rand(8192, device='xpu', dtype=torch.float32)
        norm = torch.empty_like(branch)
        updated = torch.empty_like(stream)
        widened = torch.empty_like(stream) if wide else None
        op(branch, stream, post, pre, alpha, beta, norm, updated, oracle['POST_EPS'], oracle['PRE_EPS'], mode == 'input_scale', rounded, widened)
        torch.xpu.synchronize()
        accuracy, payload = oracle['check_accuracy_xe35'](
            (branch, stream, post, pre, alpha, beta, mode == 'input_scale', rounded), norm, updated, widened)
        if wide:
            payload['widen'] = torch.empty_like(updated, device='cpu')
            for start in range(0, rows, 512):
                payload['widen'][start:start + 512].copy_(widened[start:start + 512].cpu())
                assert torch.equal(payload['widen'][start:start + 512].view(torch.int32),
                                   payload['norm'][start:start + 512].float().view(torch.int32))
        key = f'{rows}_{mode}_{int(rounded)}_{int(wide)}'
        path = output_dir / f'{key}.pt'
        torch.save(payload, path)
        record = {'case': key, 'rows': rows, 'columns': 8192, 'provider': provider,
                  'CPU_reference_all_elements': True, 'widen_bitwise_exact': True if wide else None,
                  'accuracy': accuracy, 'tensor_sha256': {name: tensor_digest(value) for name, value in payload.items()},
                  'file_sha256': digest(path), 'invocations': 1, 'official_AFTER': False}
        results.append(record)
        print(json.dumps(record), flush=True)
        (OUTPUT / f'{provider}-cases.json').write_text(json.dumps(results, indent=2) + '\n')
        del branch, stream, post, pre, alpha, beta, norm, updated, widened, payload
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)
    assert len(results) == 24


def compare():
    global torch
    import torch
    assert not torch.xpu.is_initialized()
    records = {provider: json.loads((OUTPUT / f'{provider}-cases.json').read_text()) for provider in ('historical', 'fresh')}
    assert all(len(value) == 24 for value in records.values())
    results = []
    for before, after in zip(records['historical'], records['fresh'], strict=True):
        assert before['case'] == after['case']
        payloads = []
        for provider, record in [('historical', before), ('fresh', after)]:
            path = OUTPUT / f'{provider}-outputs' / (record['case'] + '.pt')
            assert digest(path) == record['file_sha256']
            payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            assert {name: tensor_digest(value) for name, value in payload.items()} == record['tensor_sha256']
            payloads.append(payload)
        assert payloads[0].keys() == payloads[1].keys()
        output_results = {}
        for name in payloads[0]:
            before_tensor, after_tensor = payloads[0][name], payloads[1][name]
            assert before_tensor.shape == after_tensor.shape == (before['rows'], 8192)
            exact = True
            differing_elements = 0
            for start in range(0, before['rows'], 512):
                before_chunk, after_chunk = before_tensor[start:start + 512], after_tensor[start:start + 512]
                bits_type = torch.int16 if name == 'norm' else torch.int32
                chunk_exact = torch.equal(before_chunk.view(bits_type), after_chunk.view(bits_type))
                exact = exact and chunk_exact
                differing_elements += (before_chunk.view(bits_type) != after_chunk.view(bits_type)).sum().item()
                tolerance = 5e-3 if name == 'norm' else (1e-5 if name == 'stream' else 0)
                torch.testing.assert_close(after_chunk.float(), before_chunk.float(), rtol=tolerance, atol=tolerance)
            output_results[name] = {'bitwise_exact': exact, 'differing_elements': differing_elements,
                                    'elements_compared': before_tensor.numel()}
            if name == 'widen':
                assert exact, 'fulltensor SOURCE widening must be bitwise exact'
        results.append({'case': before['case'], 'outputs': output_results})
        print(json.dumps(results[-1]), flush=True)
        (OUTPUT / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
        del payloads, before_tensor, after_tensor, before_chunk, after_chunk
        gc.collect()
    assert not torch.xpu.is_initialized()
    summary = {'cases': 24, 'all_outputs_fulltensor': True, 'all_bitwise_exact': all(
        item['bitwise_exact'] for result in results for item in result['outputs'].values()),
        'strict_original_thresholds_pass': True, 'no_timing': True, 'official_AFTER': False,
        'scope': 'historical original SOURCE versus fresh icpx SOURCE compiler accuracy only'}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    scope()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', required=True, choices=['audit', 'import', 'run', 'compare'])
    parser.add_argument('--provider', choices=['historical', 'fresh'])
    args = parser.parse_args()
    if args.mode in ('import', 'run') and args.provider is None:
        parser.error('provider required')
    if args.mode == 'audit':
        print(json.dumps(audit(), indent=2), flush=True)
    elif args.mode == 'import':
        print(json.dumps(normal_import(args.provider)[3], indent=2), flush=True)
    elif args.mode == 'run':
        run(args.provider)
    else:
        compare()


if __name__ == '__main__':
    main()

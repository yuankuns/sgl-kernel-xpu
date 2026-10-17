import argparse
import ast
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import source_compiler_accuracy as source_check


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'after-pre-wave-accuracy-v4'
STAGE = ROOT / 'canonical-v4-icpx/site'
MANIFEST = ROOT / 'canonical-v4-icpx/source-binary-manifest.json'
BARRIER = ROOT / 'gpu7-after-pre-wave-accuracy-v4-grant.json'
PHASE = 'residual-genuine-after-pre-wave-accuracy-v4'
EXPECTED_MANIFEST = 'e48505e3587a14975707fc70d8d8a236b007c1922291b20a9457a8b7551bfe60'
BENCH = ROOT.parent.parent / 'pr/muse-residual-norm-xe35/benchmark/bench_fused_norm_residual.py'


def digest(path):
    return source_check.digest(path)


def provenance():
    assert digest(MANIFEST) == EXPECTED_MANIFEST
    coordinator = json.loads((ROOT / 'COORDINATOR_FULL_ICPX_AUDIT_v4.json').read_text())
    assert coordinator['verified'] is True and coordinator['version'] == 'canonical-v4-icpx'
    assert coordinator['package_manifest_sha256'] == EXPECTED_MANIFEST
    assert coordinator['default_all_actual_exit'] == coordinator['install_actual_exit'] == 0
    assert coordinator['source_postverified'] is True
    namespace = runpy.run_path(str(ROOT / 'check_canonical_package_v3.py'))
    _, stage, manifest, expected, manifest_sha = namespace['provenance']('canonical-v4-icpx')
    assert stage.resolve() == STAGE.resolve()
    assert len(expected) == coordinator['installed_and_python_files_rehashed'] == 1672
    assert len(manifest['binaries']) == coordinator['native_binary_inventory_rehashed'] == 712
    for record in manifest['binaries'].values():
        assert digest(STAGE / record['path']) == record['sha256']
    policy_sha = digest(ROOT / 'canonical-v4-icpx/compiler-policy-verified.json')
    assert policy_sha == coordinator['compiler_policy_sha256']
    assert manifest['source']['source_digest'] == coordinator['source_digest']
    before = source_check.audit()
    final = json.loads((ROOT / 'source-compiler-accuracy-v1/final-audit.json').read_text())
    assert final['terminal_actual_exit'] == 0 and final['GPU7_lock_released'] is True
    assert final['all24_full_CPU_refs_verified'] is True and final['all_actual_norm_residual_widen_bitwise_exact'] is True
    assert all(value == 0 for value in final['actual_phase_exits'].values())
    return manifest, expected, {'AFTER_manifest_sha256': manifest_sha,
        'AFTER_source_digest': manifest['source']['source_digest'], 'compiler_policy_sha256': policy_sha,
        'SOURCE_manifest_sha256': before['source_manifest_sha256'], 'SOURCE_native_sha256': before['native_sha256'],
        'AFTER_native_sha256': manifest['binaries']['libsgl-ops-sycl-FusedNormResidual.so']['sha256'],
        'coordinator_audit_sha256': digest(ROOT / 'COORDINATOR_FULL_ICPX_AUDIT_v4.json'),
        'fresh_SOURCE_cases_sha256': digest(ROOT / 'source-compiler-accuracy-v1/fresh-cases.json'),
        'normal_full_default_all_install_post_supervisor0': True, 'protected_SOURCE0': True,
        'not_clean_independent_build': True, 'build_scope': coordinator['build_scope']}


def driver_proof():
    record = {}
    for name in ('icpx', 'icx'):
        path = Path('/opt/gfx-deps/dpcpp/bin') / name
        version = subprocess.check_output([str(path), '--version'], text=True)
        assert 'Intel' in version and 'oneAPI' in version
        record[name] = {'path': str(path), 'realpath': str(path.resolve()), 'sha256': digest(path), 'version': version}
    macros = subprocess.run([record['icpx']['path'], '-dM', '-E', '-x', 'c++', '-'],
                            input='', text=True, capture_output=True, check=True).stdout
    assert '#define __INTEL_LLVM_COMPILER ' in macros
    record['IntelLLVM_predefined_macros_sha256'] = hashlib.sha256(macros.encode()).hexdigest()
    record['SYCL_runtime'] = {'path': '/opt/gfx-deps/dpcpp/lib/libsycl.so',
                             'sha256': digest(Path('/opt/gfx-deps/dpcpp/lib/libsycl.so'))}
    cache = subprocess.check_output(['/sbin/ldconfig', '-p'], text=True)
    libraries = {}
    for line in cache.splitlines():
        if any(name in line for name in ('libze_loader.so', 'libze_intel_gpu.so', 'libigc.so', 'libigdgmm.so')) and ' => ' in line:
            path = Path(line.split(' => ')[-1]).resolve()
            if path.is_file():
                libraries[str(path)] = digest(path)
    assert any('libze_intel_gpu.so' in name for name in libraries)
    record['GPU_driver_candidate_libraries_sha256'] = libraries
    return record


def normal_provider(manifest, expected):
    assert os.environ['LD_LIBRARY_PATH'].split(':')[0] == '/opt/gfx-deps/profiling_tools/lib'
    import torch
    assert not torch.xpu.is_initialized()
    assert torch.ones(4, device='cpu').sum().item() == 4
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve() == (STAGE / 'sgl_kernel/__init__.py').resolve()
    libraries = {}
    for filename in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if 'libsgl-ops-sycl-' in line or '/common_ops' in line}):
        path = Path(filename).resolve()
        relative = path.relative_to(STAGE.resolve()).as_posix()
        assert digest(path) == expected[relative]['sha256']
        libraries[relative] = digest(path)
    native = {record['path'] for record in manifest['binaries'].values()}
    required = {manifest['common_ops']} | {record['path'] for name, record in manifest['binaries'].items()
                                         if name in manifest['common_ops_dt_needed']}
    assert required <= libraries.keys() and libraries.keys() <= native
    assert len(libraries) == 712
    op = torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm_xe35
    assert 'fused_post_norm_residual_pre_norm_xe35' in str(op.default._schema)
    assert callable(sgl_kernel.fused_post_norm_residual_pre_norm_xe35)
    assert not torch.xpu.is_initialized()
    proof = {'package': str(sgl_kernel.__file__), 'new_op_schema': str(op.default._schema),
             'mapped_native_libraries': libraries, 'GPU_initialized': False,
             'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH'], 'genuine_normal_full_package': True}
    return torch, sgl_kernel.fused_post_norm_residual_pre_norm_xe35, proof


def preflight():
    manifest, expected, record = provenance()
    _, _, provider = normal_provider(manifest, expected)
    (OUTPUT / 'CPU-provider-maps.json').write_text(json.dumps(provider, indent=2) + '\n')
    (OUTPUT / 'driver-sha.json').write_text(json.dumps(driver_proof(), indent=2) + '\n')
    record.update(driver_sha256=digest(OUTPUT / 'driver-sha.json'), CPU_provider_maps_sha256=digest(OUTPUT / 'CPU-provider-maps.json'))
    (OUTPUT / 'CPU-preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


def grant():
    released = json.loads(BARRIER.read_text())
    ready = json.loads((OUTPUT / 'ready.json').read_text())
    expected = json.loads((OUTPUT / 'CPU-preflight.json').read_text())
    for name, value in expected.items():
        if name.endswith('sha256') or name == 'AFTER_source_digest':
            assert released[name] == value, name
    assert released['issuer'] == 'main coordinator user authorization'
    assert released['phase'] == PHASE and released['gpu'] == 7
    assert released['accuracy_only'] is True and released['coordinator_release'] is True
    assert released['scope'] == 'B8/ISL4096/OSL16/TP1/EP1'
    assert ready['ready_epoch'] <= released['start_epoch'] <= time.time()
    assert digest(MANIFEST) == released['AFTER_manifest_sha256']
    return released


def tensor_digest(tensor):
    result = hashlib.sha256()
    for start in range(0, tensor.shape[0], 512):
        result.update(tensor[start:start + 512].contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def accuracy():
    grant()
    manifest, expected, _ = provenance()
    global torch
    torch, invoke, proof = normal_provider(manifest, expected)
    assert os.environ.get('ZE_AFFINITY_MASK') == '7'
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if not torch.xpu.is_available():
        raise RuntimeError('XPU unavailable before Xe35 architecture query')
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch(), 'only compute capability 3.5'
    assert driver_proof() == json.loads((OUTPUT / 'driver-sha.json').read_text())
    (OUTPUT / 'runtime-normal-provider.json').write_text(json.dumps(proof, indent=2) + '\n')
    tree = ast.parse(BENCH.read_text())
    names = {'round_bfloat16_xe35', 'reference_xe35', 'check_accuracy_xe35'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('POST_EPS', 'PRE_EPS')}
    assert len(functions) == 3 and constants == {'POST_EPS': 1e-8, 'PRE_EPS': 1e-5}
    oracle = dict(constants, torch=torch, hashlib=hashlib)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BENCH), 'exec'), oracle)
    output_dir = OUTPUT / 'AFTER-outputs'
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
        invoke(branch, stream, post, pre, alpha, beta, norm, updated, constants['POST_EPS'], constants['PRE_EPS'], mode == 'input_scale', rounded, widened)
        torch.xpu.synchronize()
        accuracy, payload = oracle['check_accuracy_xe35'](
            (branch, stream, post, pre, alpha, beta, mode == 'input_scale', rounded), norm, updated, widened)
        if wide:
            payload['widen'] = torch.empty_like(updated, device='cpu')
            for start in range(0, rows, 512):
                payload['widen'][start:start + 512].copy_(widened[start:start + 512].cpu())
                assert torch.equal(payload['widen'][start:start + 512].view(torch.int32), payload['norm'][start:start + 512].float().view(torch.int32))
        key = f'{rows}_{mode}_{int(rounded)}_{int(wide)}'
        path = output_dir / f'{key}.pt'
        torch.save(payload, path)
        record = {'case': key, 'rows': rows, 'columns': 8192, 'provider': 'genuine normal AFTER canonical-v4-icpx',
                  'CPU_reference_all_elements': True, 'accuracy': accuracy, 'widen_bitwise_exact': True if wide else None,
                  'tensor_sha256': {name: tensor_digest(value) for name, value in payload.items()},
                  'file_sha256': digest(path), 'production_case_invocations': 1, 'performance_approved': False}
        results.append(record)
        (OUTPUT / 'AFTER-cases.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record), flush=True)
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
    before = json.loads((ROOT / 'source-compiler-accuracy-v1/fresh-cases.json').read_text())
    after = json.loads((OUTPUT / 'AFTER-cases.json').read_text())
    assert len(before) == len(after) == 24
    results = []
    for source, port in zip(before, after, strict=True):
        assert source['case'] == port['case']
        source_file = ROOT / 'source-compiler-accuracy-v1/fresh-outputs' / (source['case'] + '.pt')
        port_file = OUTPUT / 'AFTER-outputs' / (port['case'] + '.pt')
        assert digest(source_file) == source['file_sha256'] and digest(port_file) == port['file_sha256']
        source_outputs = torch.load(source_file, map_location='cpu', mmap=True, weights_only=True)
        port_outputs = torch.load(port_file, map_location='cpu', mmap=True, weights_only=True)
        assert source_outputs.keys() == port_outputs.keys()
        outputs = {}
        for name in source_outputs:
            source_tensor, port_tensor = source_outputs[name], port_outputs[name]
            assert source_tensor.shape == port_tensor.shape == (source['rows'], 8192)
            assert tensor_digest(source_tensor) == source['tensor_sha256'][name]
            assert tensor_digest(port_tensor) == port['tensor_sha256'][name]
            differing_elements = 0
            for start in range(0, source['rows'], 512):
                source_chunk, port_chunk = source_tensor[start:start + 512], port_tensor[start:start + 512]
                bits = torch.int16 if name == 'norm' else torch.int32
                differing_elements += (source_chunk.view(bits) != port_chunk.view(bits)).sum().item()
                tolerance = 5e-3 if name == 'norm' else 1e-5
                torch.testing.assert_close(port_chunk.float(), source_chunk.float(), rtol=tolerance, atol=tolerance)
            outputs[name] = {'bitwise_exact': differing_elements == 0, 'differing_elements': differing_elements,
                             'elements_compared': source_tensor.numel()}
        results.append({'case': source['case'], 'outputs': outputs})
        (OUTPUT / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1]), flush=True)
        del source_outputs, port_outputs, source_tensor, port_tensor, source_chunk, port_chunk
        gc.collect()
    assert all(item['bitwise_exact'] for result in results for item in result['outputs'].values())
    assert not torch.xpu.is_initialized()
    print(json.dumps({'cases': 24, 'all_actual_outputs_bitwise_exact_fresh_SOURCE_to_genuine_AFTER': True,
                      'performance_approved': False}), flush=True)


def main():
    source_check.scope()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', required=True, choices=['preflight', 'accuracy', 'compare', 'post'])
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'accuracy':
        accuracy()
    elif args.mode == 'compare':
        compare()
    else:
        _, _, record = provenance()
        assert driver_proof() == json.loads((OUTPUT / 'driver-sha.json').read_text())
        (OUTPUT / 'post-protected-source.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

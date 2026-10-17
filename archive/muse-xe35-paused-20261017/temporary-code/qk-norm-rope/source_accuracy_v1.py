import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARTIFACT = Path(__file__).resolve().parent
RUN = ARTIFACT / 'source-accuracy-v1'
VERSION = 'qk-source-accuracy-v1'
GRANT = ARTIFACT / 'source-accuracy-v1-grant.json'
ROLES = ('historical-source', 'fresh-source', 'fresh-preopt')


def scope():
    for name, expected in {
        'BATCHES': '8', 'BATCH_SIZE': '8', 'BATCH': '8', 'SERVING_BATCH': '8',
        'SERVING_BATCH_SIZE': '8', 'SGLANG_BATCH_SIZE': '8',
        'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1',
    }.items():
        if os.environ.get(name, expected) != expected:
            raise SystemExit(f'scope error: {name} must be {expected}; B8 only')
    for name in ('TIMING', 'BENCHMARK', 'WARMUP', 'REPEATS', 'SAMPLES', 'GRAPH', 'PERF',
                 'QK_TIMING', 'QK_BENCHMARK', 'QK_WARMUP', 'QK_REPEATS', 'QK_SAMPLES'):
        if name in os.environ:
            raise SystemExit(f'accuracy-only rejects timing override {name}')


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def prepare():
    from canonical_check import inventory, local_path
    from fresh_native_check import verify_provider

    assert not RUN.exists() and not GRANT.exists(), 'New accuracy grant/run only'
    source = verify_provider('optimized-source')
    preopt = verify_provider('preopt')
    history = ARTIFACT.parents[2] / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
    original = json.loads((history / 'provenance.json').read_text())['after']
    historical_manifest = history / 'after/site-manifest.json'
    assert inventory.digest_file(historical_manifest) == original['site_manifest_sha256']
    historical_site = history / 'after/site'
    records = json.loads(historical_manifest.read_text())
    expected = {str(Path(record['path']).relative_to('after/site')): record['sha256'] for record in records}
    actual = {str(path.relative_to(historical_site)): inventory.digest_file(path)
              for path in historical_site.rglob('*') if path.is_file()}
    assert actual == expected
    assert inventory.digest_file(historical_site / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so') == original['qk_library_sha256']
    providers = {
        'historical-source': {'ref': original['commit'], 'site': str(historical_site),
             'manifest': str(historical_manifest), 'manifest_sha256': original['site_manifest_sha256'],
             'manifest_format': 'historical-list', 'native_sha256': original['qk_library_sha256'],
             'scope': 'Historical genuine optimized SOURCE complete normal package; GNU host plus icx SYCL'},
    }
    for role, label, record in (('fresh-source', 'source', source), ('fresh-preopt', 'preopt', preopt)):
        providers[role] = {
            'ref': record[f'{label}_commit'], 'site': record[f'{label}_site'],
            'manifest': record[f'{label}_manifest'], 'manifest_sha256': record[f'{label}_manifest_sha256'],
            'manifest_format': record[f'{label}_manifest_format'],
            'native_sha256': record[f'{label}_qk_library_sha256'],
            'scope': record[f'{label}_native_build_scope'],
        }
    RUN.mkdir()
    for role in ROLES:
        (RUN / role).mkdir()
    save(RUN / 'provenance.json', {'version': VERSION, 'gpu': 6, 'cpus': '32-47',
         'scope': {'batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'bf16'},
         'providers': providers, 'precision_only': True, 'PORT_provider': False,
         'timing_graph_events_warmup_allowed': False,
         'historical_preopt3508_actual_exit1_retained': True,
         'fresh_preopt_violations_not_required_to_equal3508': True})


def provider(role):
    from canonical_check import inventory, local_path

    provenance = json.loads((RUN / 'provenance.json').read_text())
    assert provenance['version'] == VERSION and provenance['precision_only'] is True
    record = provenance['providers'][role]
    site = local_path(record['site'])
    manifest = local_path(record['manifest'])
    assert inventory.digest_file(manifest) == record['manifest_sha256']
    contents = json.loads(manifest.read_text())
    if record['manifest_format'] == 'historical-list':
        expected = {str(Path(item['path']).relative_to('after/site')): item['sha256'] for item in contents}
    else:
        assert record['manifest_format'] == 'relative-sha256-map'
        expected = contents
    actual = {str(path.relative_to(site)): inventory.digest_file(path)
              for path in site.rglob('*') if path.is_file()}
    assert actual == expected, 'Whole normal package inventory mismatch'
    assert actual['sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'] == record['native_sha256']
    return record, site, expected


def mapped_audit(role):
    from canonical_check import inventory
    import torch
    import sgl_kernel

    record, site, expected = provider(role)
    assert Path(os.environ['PYTHONPATH']).resolve() == site.resolve()
    assert Path(sgl_kernel.__file__).resolve().is_relative_to(site.resolve())
    assert Path(sgl_kernel.common_ops.__file__).resolve().is_relative_to(site.resolve())
    assert callable(sgl_kernel.fused_qk_norm_rope)
    assert hasattr(torch.ops.sgl_kernel, 'fused_qk_norm_rope')
    for name, module in tuple(sys.modules.items()):
        if name == 'sgl_kernel' or name.startswith('sgl_kernel.'):
            origin = getattr(module, '__file__', None)
            if origin:
                assert Path(origin).resolve().is_relative_to(site.resolve()), origin
    mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5])
        if not (path.name.startswith('libsgl-ops-') or path.name.startswith('common_ops.')):
            continue
        assert path.resolve().is_relative_to(site.resolve()), f'Foreign native mapped: {path}'
        relative = str(path.relative_to(site))
        assert relative in expected and inventory.digest_file(path) == expected[relative]
        mapped[str(path)] = expected[relative]
    native = site / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'
    assert mapped[str(native)] == record['native_sha256']
    assert str(Path(sgl_kernel.common_ops.__file__)) in mapped
    return {'role': role, 'normal_package': sgl_kernel.__file__, 'common_ops': sgl_kernel.common_ops.__file__,
            'all_mapped_native_hashes_verified': mapped, 'whole_site_files_verified': len(expected),
            'manifest_sha256': record['manifest_sha256'], 'native_sha256': record['native_sha256']}


def command(arguments, prefix, environment):
    prefix.with_suffix('.command.json').write_text(json.dumps(arguments) + '\n')
    with prefix.with_suffix('.log').open('w') as output:
        child = subprocess.Popen(arguments, env=environment, stdout=output, stderr=subprocess.STDOUT)
        save(prefix.with_suffix('.pid.json'), {'pid': child.pid, 'ppid': os.getpid(), 'argv': arguments})
        actual = child.wait()
    prefix.with_suffix('.exit').write_text(f'{actual}\n')
    return actual


def cpu_import(role):
    import torch

    assert not torch.xpu.is_initialized()
    result = mapped_audit(role)
    assert not torch.xpu.is_initialized()
    result['GPU_initialized'] = False
    save(RUN / role / 'cpu-import-audit.json', result)
    print(json.dumps(result), flush=True)


def cpu_preflight():
    from canonical_check import local_path

    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    for role in ROLES:
        record, site, expected = provider(role)
        environment = {**os.environ, 'PYTHONPATH': str(site), 'PYTHONDONTWRITEBYTECODE': '1'}
        actual = command([sys.executable, str(Path(__file__).resolve()), '--mode', 'cpu-import', '--role', role],
                         RUN / role / 'cpu-import', environment)
        assert actual == 0, f'{role} CPU normal import preflight failed'


def publish_grant():
    assert not GRANT.exists()
    provenance = json.loads((RUN / 'provenance.json').read_text())
    for role in ROLES:
        assert (RUN / role / 'cpu-import.exit').read_text().strip() == '0'
        audit = json.loads((RUN / role / 'cpu-import-audit.json').read_text())
        assert audit['GPU_initialized'] is False
        assert audit['native_sha256'] == provenance['providers'][role]['native_sha256']
    save(GRANT, {'version': VERSION, 'grant_kind': 'SOURCE-accuracy-only', 'gpu': 6, 'cpus': '32-47',
         'authorization': 'Explicit user/main GPU6 SOURCE accuracy-only grant in current conversation',
         'coordinator_release': True, 'start_epoch': time.time() + 45,
         'providers': provenance['providers'], 'scope': provenance['scope'],
         'timing_graph_events_warmup_allowed': False, 'PORT_comparison_authorized': False,
         'W4_lease_transfer_authorized': False})
    print(GRANT.read_text(), flush=True)


def verify_grant():
    grant = json.loads(GRANT.read_text())
    provenance = json.loads((RUN / 'provenance.json').read_text())
    assert grant['version'] == VERSION and grant['grant_kind'] == 'SOURCE-accuracy-only'
    assert grant['gpu'] == 6 and grant['cpus'] == '32-47' and grant['coordinator_release'] is True
    assert grant['providers'] == provenance['providers'] and grant['scope'] == provenance['scope']
    assert grant['timing_graph_events_warmup_allowed'] is False
    assert grant['PORT_comparison_authorized'] is False and grant['W4_lease_transfer_authorized'] is False
    assert os.environ['ZE_AFFINITY_MASK'] == '6'
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    for role in ROLES:
        provider(role)
    return grant


def accuracy(role, tokens):
    verify_grant()
    import torch
    import sgl_kernel

    audit_before = mapped_audit(role)
    assert hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU unavailable'
    architecture = torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device())
    assert tuple(architecture) == (3, 5), f'Xe35-only, actual {architecture}'
    torch.manual_seed(42)
    dimension, query_heads, key_heads, tail_heads = 64, 128, 16, 144
    epsilon, base = 1e-5, 500000.0
    scale = 43.7840518911 / dimension ** 0.5
    qkv = torch.randn(tokens, (query_heads + key_heads + tail_heads) * dimension,
                      device='xpu:0', dtype=torch.bfloat16)
    original = qkv.cpu()
    query_weight = torch.full((dimension,), scale, device='xpu:0', dtype=torch.bfloat16)
    key_weight = torch.ones(dimension, device='xpu:0', dtype=torch.bfloat16)
    positions = torch.arange(tokens, device='xpu:0', dtype=torch.int32) % 4096
    inverse_frequency = base ** (-torch.arange(0, dimension, 2, dtype=torch.float64, device='cpu') / dimension)
    sgl_kernel.fused_qk_norm_rope(qkv, query_heads, key_heads, tail_heads, dimension, epsilon,
                                query_weight, key_weight, base, False, positions, rotary_dim=dimension)
    torch.xpu.synchronize()
    actual = qkv.cpu()
    positions_cpu = positions.cpu()
    query_weight_cpu, key_weight_cpu = query_weight.float().cpu(), key_weight.float().cpu()
    violations, elements, maximum_error = 0, 0, 0.0
    for start in range(0, tokens, 512):
        rows = original[start:start + 512].reshape(-1, query_heads + key_heads + tail_heads, dimension).float()
        query = rows[:, :query_heads]
        key = rows[:, query_heads:query_heads + key_heads]
        query = query * torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + epsilon) * query_weight_cpu
        key = key * torch.rsqrt(key.square().mean(dim=-1, keepdim=True) + epsilon) * key_weight_cpu
        angles = positions_cpu[start:start + 512].float()[:, None] * inverse_frequency.float()[None, :]
        cosine, sine = angles.cos()[:, None, :], angles.sin()[:, None, :]

        def rotate(values):
            even, odd = values[..., ::2], values[..., 1::2]
            return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)

        expected = torch.cat((rotate(query), rotate(key)), dim=1)
        got = actual[start:start + 512, :(query_heads + key_heads) * dimension].reshape_as(expected)
        difference = (got.float() - expected).abs()
        violations += ((~torch.isfinite(got)) | (~torch.isfinite(expected))
                       | (difference > 5e-3 + 5e-3 * expected.abs())).sum().item()
        elements += difference.numel()
        maximum_error = max(maximum_error, difference.max().item())
    tail_equal = torch.equal(actual[:, (query_heads + key_heads) * dimension:].view(torch.uint16),
                             original[:, (query_heads + key_heads) * dimension:].view(torch.uint16))
    output = RUN / role / f'accuracy-t{tokens}'
    torch.save({'original': original, 'actual': actual, 'positions': positions_cpu,
                'query_weight': query_weight_cpu, 'key_weight': key_weight_cpu}, output.with_suffix('.pt'))
    audit_after = mapped_audit(role)
    result = {'role': role, 'tokens': tokens, 'dtype': 'bf16', 'production': True,
              'heads': [128, 16, 144], 'head_dim': 64, 'atol': 5e-3, 'rtol': 5e-3,
              'reference': 'independent CPU FP32 with FP64 negative-pow frequency rounded to FP32; original reference algorithm unchanged',
              'elements': elements, 'violations': violations, 'max_abs': maximum_error, 'tail_equal': tail_equal,
              'input_sha256': hashlib.sha256(original.view(torch.uint16).numpy().tobytes()).hexdigest(),
              'output_sha256': hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
              'raw_all_QKV_input_output': str(output.with_suffix('.pt')),
              'mapped_before': audit_before, 'mapped_after': audit_after,
              'native_invocations': 1, 'timing_graph_events_warmup_executed': False}
    save(output.with_suffix('.json'), result)
    print(json.dumps(result), flush=True)
    assert violations == 0 and tail_equal, 'Strict production CPU reference failed'


def suite():
    grant = verify_grant()
    assert grant['start_epoch'] > time.time(), 'Fresh grant epoch required, not canonical/old grant'
    save(RUN / 'gpu-supervisor.json', {'pid': os.getpid(), 'ppid': os.getppid(),
         'gpu': 6, 'affinity': sorted(os.sched_getaffinity(0)), 'grant_version': VERSION})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    records = {}
    for tokens in (8, 32768):
        for role in ROLES:
            record, site, expected = provider(role)
            environment = {**os.environ, 'PYTHONPATH': str(site), 'PYTHONDONTWRITEBYTECODE': '1'}
            output = RUN / role / f'accuracy-t{tokens}'
            actual = command([sys.executable, str(Path(__file__).resolve()), '--mode', 'case',
                              '--role', role, '--tokens', str(tokens)], output, environment)
            result = json.loads(output.with_suffix('.json').read_text())
            records[f'{role}-t{tokens}'] = {'actual_exit': actual, 'result': result}
            save(RUN / 'phase-results.json', records)
            if role != 'fresh-preopt':
                assert actual == 0 and result['violations'] == 0 and result['tail_equal']
            else:
                assert result['tail_equal'] and actual == (1 if result['violations'] else 0)
            time.sleep(2)
    save(RUN / 'phase-results.json', records)
    compare()


def compare():
    import torch

    results = {}
    for tokens in (8, 32768):
        historical = torch.load(RUN / 'historical-source' / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
        fresh = torch.load(RUN / 'fresh-source' / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
        preopt = torch.load(RUN / 'fresh-preopt' / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
        input_equal = torch.equal(historical['original'].view(torch.uint16), fresh['original'].view(torch.uint16))
        preopt_input_equal = torch.equal(historical['original'].view(torch.uint16), preopt['original'].view(torch.uint16))
        output_equal = torch.equal(historical['actual'].view(torch.uint16), fresh['actual'].view(torch.uint16))
        bitwise_differences = 0
        for start in range(0, tokens, 512):
            bitwise_differences += (historical['actual'][start:start + 512].view(torch.uint16)
                                   != fresh['actual'][start:start + 512].view(torch.uint16)).sum().item()
        tail = 144 * 64
        tail_equal = torch.equal(historical['actual'][:, tail:].view(torch.uint16), fresh['actual'][:, tail:].view(torch.uint16))
        for key in ('positions', 'query_weight', 'key_weight'):
            assert torch.equal(historical[key], fresh[key]) and torch.equal(historical[key], preopt[key])
        results[str(tokens)] = {'input_bitwise': input_equal, 'preopt_input_bitwise': preopt_input_equal,
                               'all_QKV_output_bitwise': output_equal, 'all_QKV_bitwise_differences': bitwise_differences,
                               'tail_bitwise': tail_equal, 'compared_output_elements': historical['actual'].numel()}
        save(RUN / 'source-equivalence.json', results)
        print(json.dumps({str(tokens): results[str(tokens)]}), flush=True)
        assert input_equal and preopt_input_equal and output_equal and tail_equal
        del historical, fresh, preopt
    save(RUN / 'source-equivalence.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    scope()
    parser = argparse.ArgumentParser(description='Accuracy-only B8 SOURCE history/fresh/preopt; no timing options')
    parser.add_argument('--mode', choices=('prepare', 'cpu-import', 'cpu-preflight', 'publish-grant', 'suite', 'case'), required=True)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--tokens', type=int, choices=(8, 32768))
    arguments = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '8'
    if arguments.mode == 'prepare':
        prepare()
    elif arguments.mode == 'cpu-import':
        assert arguments.role and arguments.tokens is None
        cpu_import(arguments.role)
    elif arguments.mode == 'cpu-preflight':
        cpu_preflight()
    elif arguments.mode == 'publish-grant':
        publish_grant()
    elif arguments.mode == 'suite':
        suite()
    else:
        assert arguments.role and arguments.tokens
        accuracy(arguments.role, arguments.tokens)

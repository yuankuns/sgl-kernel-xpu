import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ARTIFACT = RUN.parent
sys.path.insert(0, str(ARTIFACT))
VERSION = 'normal-v4-accuracy-only-v1'
SOURCE = ARTIFACT / 'source-accuracy-v1/fresh-source'
GRANT = RUN / 'accuracy-only-grant.json'


def scope():
    from source_accuracy_v1 import scope as original_scope

    original_scope()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def digest(path):
    from canonical_check import inventory

    return inventory.digest_file(path)


def provenance():
    return json.loads((RUN / 'provenance.json').read_text())


def prepare():
    from canonical_check import check

    assert not (RUN / 'provenance.json').exists() and not GRANT.exists()
    current = check('canonical-v4-icpx')
    assert current['canonical_manifest_sha256'] == 'f07b42b02097669331ae3e44f071068ad5bc3e7b5a84913172b450d1f504cdc9'
    source_audit_path = ARTIFACT / 'source-accuracy-v1/final-evidence-audit.json'
    source_audit = json.loads(source_audit_path.read_text())
    assert source_audit['actual_exit'] == 0 and source_audit['SOURCE_equivalence_accuracy'].startswith('PASS:')
    raw = {}
    for tokens in (8, 32768):
        for suffix in ('.pt', '.json', '.exit'):
            relative = f'fresh-source/accuracy-t{tokens}{suffix}'
            path = ARTIFACT / 'source-accuracy-v1' / relative
            assert digest(path) == source_audit['raw_files'][relative]['sha256']
            raw[relative] = source_audit['raw_files'][relative]
        case = json.loads((SOURCE / f'accuracy-t{tokens}.json').read_text())
        assert case['violations'] == 0 and case['tail_equal']
        assert case['atol'] == case['rtol'] == 5e-3
        assert (SOURCE / f'accuracy-t{tokens}.exit').read_text().strip() == '0'
    current.update({'version': VERSION, 'gpu': 6, 'cpus': '32-47',
        'scope': {'batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'bf16'},
        'source_raw_files': raw, 'source_accuracy_audit_sha256': digest(source_audit_path),
        'reviewed_CPU_reference_script_sha256': digest(ARTIFACT / 'source_accuracy_v1.py'),
        'driver_sha256': digest(RUN / 'driver.py'), 'launcher_sha256': digest(RUN / 'launch.sh'),
        'accuracy_only': True, 'formal_performance_wave': False,
        'timing_events_graphs_warmup_allowed': False, 'preopt_rerun': False})
    save(RUN / 'provenance.json', current)


def verify_inputs():
    from canonical_check import local_path

    record = provenance()
    assert record['version'] == VERSION and record['accuracy_only']
    assert digest(RUN / 'driver.py') == record['driver_sha256']
    assert digest(RUN / 'launch.sh') == record['launcher_sha256']
    assert digest(ARTIFACT / 'source_accuracy_v1.py') == record['reviewed_CPU_reference_script_sha256']
    for relative, expected in record['source_raw_files'].items():
        assert digest(ARTIFACT / 'source-accuracy-v1' / relative) == expected['sha256']
    assert digest(local_path(record['canonical_manifest'])) == record['canonical_manifest_sha256']
    assert digest(local_path(record['source_manifest'])) == record['source_manifest_sha256']
    assert digest(local_path(record['compiler_policy_proof']['compiler_policy_file'])) == record['compiler_policy_proof']['compiler_policy_sha256']
    return record


def normal_audit():
    from canonical_check import inventory, local_path
    from runtime_check import audit

    record = verify_inputs()
    site = local_path(record['port_site']).resolve()
    sys.path.insert(0, str(site))
    os.environ['PYTHONPATH'] = str(site)
    mapped = audit(record, 'port', site)
    manifest = json.loads(local_path(record['canonical_manifest']).read_text())
    inventory.verify_inventory(site, manifest['installed_files'] | manifest['python_snapshot_files'])
    assert set(mapped) == {str(site / item['path']) for item in manifest['binaries'].values()}
    assert len(mapped) == 712
    return {'whole_site_files_verified': len(manifest['installed_files'] | manifest['python_snapshot_files']),
            'all_mapped_native_hashes_verified': mapped, 'mapped_native_count': len(mapped)}


def cpu_preflight():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    import torch

    assert not torch.xpu.is_initialized()
    mapped = normal_audit()
    import sgl_kernel

    assert callable(sgl_kernel.fused_qk_norm_rope_xe35)
    assert 'fused_qk_norm_rope_xe35' in str(torch.ops.sgl_kernel.fused_qk_norm_rope_xe35.default._schema)
    for tokens in (8, 32768):
        data = torch.load(SOURCE / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
        assert set(data) == {'original', 'actual', 'positions', 'query_weight', 'key_weight'}
        assert data['original'].shape == data['actual'].shape == (tokens, 288 * 64)
        assert data['original'].dtype == data['actual'].dtype == torch.bfloat16
        assert torch.equal(data['positions'], torch.arange(tokens, dtype=torch.int32) % 4096)
        assert torch.equal(data['actual'][:, 144 * 64:].view(torch.uint16), data['original'][:, 144 * 64:].view(torch.uint16))
        for name in ('query_weight', 'key_weight'):
            assert torch.equal(data[name], data[name].to(torch.bfloat16).float())
        del data
    assert not torch.xpu.is_initialized()
    save(RUN / 'cpu-preflight.json', {'actual_exit': 0, 'pid': os.getpid(),
        'XPU_initialized': False, 'normal_package': mapped, 'source_shapes_checked': [8, 32768]})
    print('CPU preflight0: genuine full normal package / 712 maps / immutable saved SOURCE inputs / XPU uninitialized', flush=True)


def publish_grant():
    record = verify_inputs()
    assert not GRANT.exists()
    assert (RUN / 'cpu-preflight.exit').read_text().strip() == '0'
    assert json.loads((RUN / 'cpu-preflight.json').read_text())['XPU_initialized'] is False
    save(GRANT, {'version': VERSION, 'grant_kind': 'genuine-normal-v4-accuracy-only',
        'authorization': 'Explicit coordinator/user independent current-full AFTER accuracy-only authorization; NOT formal performance eight-card wave',
        'gpu': 6, 'cpus': '32-47', 'coordinator_release': True, 'start_epoch': time.time() + 45,
        'source_digest': record['ready_source_sha256'],
        'package_manifest_sha256': record['canonical_manifest_sha256'],
        'compiler_policy_sha256': record['compiler_policy_proof']['compiler_policy_sha256'],
        'SOURCE_manifest_sha256': record['source_manifest_sha256'],
        'source_raw_files': record['source_raw_files'], 'driver_sha256': record['driver_sha256'],
        'launcher_sha256': record['launcher_sha256'], 'scope': record['scope'],
        'timing_events_graphs_warmup_allowed': False, 'formal_performance_wave': False,
        'preopt_rerun': False, 'W4_lease_transfer_authorized': False})
    print(GRANT.read_text(), flush=True)


def verify_grant():
    record = verify_inputs()
    grant = json.loads(GRANT.read_text())
    assert grant['version'] == VERSION and grant['grant_kind'] == 'genuine-normal-v4-accuracy-only'
    assert grant['gpu'] == 6 and grant['cpus'] == '32-47' and grant['coordinator_release']
    for field, expected in {'source_digest': record['ready_source_sha256'],
        'package_manifest_sha256': record['canonical_manifest_sha256'],
        'compiler_policy_sha256': record['compiler_policy_proof']['compiler_policy_sha256'],
        'SOURCE_manifest_sha256': record['source_manifest_sha256'],
        'driver_sha256': record['driver_sha256'], 'launcher_sha256': record['launcher_sha256'],
        'source_raw_files': record['source_raw_files'], 'scope': record['scope']}.items():
        assert grant[field] == expected
    assert not grant['timing_events_graphs_warmup_allowed'] and not grant['formal_performance_wave']
    assert not grant['preopt_rerun'] and not grant['W4_lease_transfer_authorized']
    assert os.environ['ZE_AFFINITY_MASK'] == '6'
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    return grant


def cpu_reference(original, actual, positions_cpu, query_weight_cpu, key_weight_cpu):
    import torch

    tokens = original.shape[0]
    dimension, query_heads, key_heads, tail_heads = 64, 128, 16, 144
    epsilon, base = 1e-5, 500000.0
    inverse_frequency = base ** (-torch.arange(0, dimension, 2, dtype=torch.float64, device='cpu') / dimension)
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
    return violations, elements, maximum_error


def case(tokens):
    verify_grant()
    import torch

    mapped_before = normal_audit()
    import sgl_kernel

    assert hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU unavailable'
    architecture = tuple(torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device()))
    assert architecture == (3, 5), f'Xe35-only: actual {architecture}'
    source = torch.load(SOURCE / f'accuracy-t{tokens}.pt', map_location='cpu', weights_only=True)
    original = source['original']
    qkv = original.to('xpu:0')
    query_weight = source['query_weight'].to(device='xpu:0', dtype=torch.bfloat16)
    key_weight = source['key_weight'].to(device='xpu:0', dtype=torch.bfloat16)
    positions = source['positions'].to('xpu:0')
    assert torch.equal(qkv.cpu().view(torch.uint16), original.view(torch.uint16))
    assert torch.equal(query_weight.float().cpu(), source['query_weight'])
    assert torch.equal(key_weight.float().cpu(), source['key_weight'])
    assert torch.equal(positions.cpu(), source['positions'])
    torch.xpu.synchronize()
    time.sleep(2)
    sgl_kernel.fused_qk_norm_rope_xe35(qkv, 128, 16, 144, 64, 1e-5, query_weight, key_weight,
                                    500000.0, False, positions, rotary_dim=64)
    torch.xpu.synchronize()
    actual = qkv.cpu()
    violations, elements, maximum_error = cpu_reference(original, actual, source['positions'],
                                                       source['query_weight'], source['key_weight'])
    difference_count = int((actual.view(torch.uint16) != source['actual'].view(torch.uint16)).sum().item())
    tail_equal = torch.equal(actual[:, 144 * 64:].view(torch.uint16), original[:, 144 * 64:].view(torch.uint16))
    prefix = RUN / f'accuracy-t{tokens}'
    torch.save({'original': original, 'actual': actual, 'positions': source['positions'],
                'query_weight': source['query_weight'], 'key_weight': source['key_weight']}, prefix.with_suffix('.pt'))
    result = {'version': VERSION, 'tokens': tokens, 'dtype': 'bf16', 'production': True,
        'heads': [128, 16, 144], 'head_dim': 64, 'atol': 5e-3, 'rtol': 5e-3,
        'violations': violations, 'elements': elements, 'max_abs': maximum_error,
        'tail_equal': tail_equal, 'full_QKV_SOURCE_output_bitwise_differences': difference_count,
        'full_QKV_output_elements': actual.numel(), 'full_SOURCE_input_bitwise': True,
        'input_sha256': hashlib.sha256(original.view(torch.uint16).numpy().tobytes()).hexdigest(),
        'output_sha256': hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
        'mapped_before': mapped_before, 'mapped_after': normal_audit(), 'architecture': architecture,
        'actual_registered_op': 'sgl_kernel.fused_qk_norm_rope_xe35', 'native_invocations': 1,
        'timing_events_graphs_warmup_executed': False}
    save(prefix.with_suffix('.json'), result)
    print(json.dumps({name: value for name, value in result.items() if not name.startswith('mapped_')}), flush=True)
    assert violations == 0 and tail_equal and difference_count == 0, 'Strict normal AFTER reference/SOURCE full-QKV equivalence failed'


def suite():
    grant = verify_grant()
    assert grant['start_epoch'] > time.time(), 'Independent fresh accuracy grant required'
    save(RUN / 'gpu-supervisor.json', {'pid': os.getpid(), 'ppid': os.getppid(),
        'gpu': 6, 'cpus': sorted(os.sched_getaffinity(0)), 'version': VERSION})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    phases = {}
    for tokens in (8, 32768):
        prefix = RUN / f'accuracy-t{tokens}'
        arguments = [sys.executable, str(RUN / 'driver.py'), '--mode', 'case', '--tokens', str(tokens)]
        save(prefix.with_suffix('.command.json'), {'argv': arguments, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'ZE_AFFINITY_MASK': os.environ['ZE_AFFINITY_MASK']})
        with prefix.with_suffix('.log').open('w') as output:
            child = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT)
            save(prefix.with_suffix('.pid.json'), {'pid': child.pid, 'ppid': os.getpid()})
            status = child.wait()
        prefix.with_suffix('.exit').write_text(f'{status}\n')
        phases[str(tokens)] = {'actual_exit': status, 'result': json.loads(prefix.with_suffix('.json').read_text()) if prefix.with_suffix('.json').exists() else None}
        save(RUN / 'phase-results.json', phases)
        assert status == 0, f'T{tokens} actual phase exit{status}; preserve failure, no retry'
    print('Normal v4 accuracy-only both BF16 production cases passed; no performance measured', flush=True)


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'cpu-preflight', 'publish-grant', 'suite', 'case'), required=True)
    parser.add_argument('--tokens', type=int, choices=(8, 32768))
    arguments = parser.parse_args()
    if arguments.mode == 'case':
        assert arguments.tokens is not None
        case(arguments.tokens)
    else:
        assert arguments.tokens is None
        {'prepare': prepare, 'cpu-preflight': cpu_preflight, 'publish-grant': publish_grant, 'suite': suite}[arguments.mode]()


if __name__ == '__main__':
    main()

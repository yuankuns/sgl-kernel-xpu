import argparse
import ast
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
sys.path.insert(0, str(ARTIFACT / 'normal-v4-accuracy-only-v1'))
from driver import cpu_reference
from canonical_check import inventory, local_path, verify_reviewed_freeze

VERSION = 'normal-v4-model-config-accuracy-v2'
FRAMEWORK = local_path('/home/syk/sglang/python').resolve()
GRANT = RUN / 'accuracy-only-grant.json'
ROLES = ('source', 'port')


def scope():
    from source_accuracy_v1 import scope as original_scope

    original_scope()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def digest(path):
    return inventory.digest_file(path)


def record():
    return json.loads((RUN / 'provenance.json').read_text())


def prepare():
    from fresh_native_check import verify_provider

    assert not (RUN / 'provenance.json').exists() and not GRANT.exists()
    ready = json.loads((ARTIFACT / 'normal-fullready-v4.json').read_text())
    assert ready['CPU_fullready_actual_exit'] == 0 and ready['complete_native_graph_mapped_and_SHA_verified'] == 712
    canonical = ARTIFACT / 'canonical-v4-icpx'
    manifest = canonical / 'source-binary-manifest.json'
    payload = json.loads(manifest.read_text())
    assert digest(manifest) == ready['canonical_manifest_sha256'] == 'f07b42b02097669331ae3e44f071068ad5bc3e7b5a84913172b450d1f504cdc9'
    for phase in ('configure', 'default-all-build', 'supervisor'):
        assert (canonical / f'{phase}.exit').read_text().strip() == '0'
    assert payload['install_actual_exit'] == 0
    assert json.loads((canonical / 'source-readonly-postverify.json').read_text())['verified']
    frozen, reviewed_path = verify_reviewed_freeze()
    assert frozen['source_digest'] == ready['source_digest']
    for name, value in frozen['files'].items():
        assert digest(ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35' / name) == value['sha256']
    assert digest(canonical / 'compiler-policy-verified.json') == ready['compiler_policy_sha256']
    policy = json.loads((canonical / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM'
    assert all(Path(policy[name]).name == 'icpx' for name in ('cxx', 'sycl', 'host'))
    assert Path(policy['c']).name == 'icx'
    source = verify_provider('optimized-source')
    derived_root = local_path('/home/syk/intel_workspace/model_enablement/profiling/results/profile_muse_spark_l3_tp1_fmha-fbd1d80-20260910_b8_i4096_o16_serving')
    derived = json.loads((derived_root / 'buckets.json').read_text())
    summary = json.loads((derived_root / 'profile-summary.jsonl').read_text().splitlines()[0])
    assert summary['random_input_len'] == 4096 and summary['random_output_len'] == 16
    assert summary['completed'] == summary['max_concurrency'] == 8
    assert summary['server_info']['disable_radix_cache'] and summary['server_info']['speculative_algorithm'] is None
    assert '001 step[EXTEND bs=3 toks=12288]' in derived['steps']
    assert '002 step[EXTEND bs=5 toks=20480]' in derived['steps']
    model_root = FRAMEWORK.parent / 'sglang_995c774/sglang_meta_common/sglang_meta_common'
    config_path = FRAMEWORK.parent / 'artifacts/muse-runtime-20261017/weights/Muse-Spark-1.2-820B-A42B-MXFP4-L3/config.json'
    config = json.loads(config_path.read_text())
    model_tree = ast.parse((model_root / 'models/muse_spark.py').read_text())
    attention = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == 'MuseSpark1xAttention')
    scaling = next(node for node in ast.walk(attention) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == 'scale_query_by' for target in node.targets))
    model_call = next(node for node in ast.walk(attention) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'fused_qk_norm_rope')
    model_keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in model_call.keywords if keyword.arg != 'rotary_dim'}
    parameters = {'num_heads_q': config['num_attention_heads'], 'num_heads_k': config['num_key_value_heads'],
        'tail_heads': config['num_key_value_heads'] + config['num_attention_heads'], 'head_dim': config['head_dim'],
        'rotary_dim': config['head_dim'], 'norm_eps': config['rms_norm_eps'], 'rope_theta': config['rope_theta'],
        'q_scaling_constant': ast.literal_eval(scaling.value.orelse.left), 'is_neox': ast.literal_eval(model_call.args[9]),
        **model_keywords}
    assert config['use_attn_o_gate'] and config['use_qk_norm'] and not config['use_sqrt_attn_scaling']
    assert parameters['num_heads_q'] == 128 and parameters['num_heads_k'] == 16 and parameters['head_dim'] == 64
    assert parameters['norm_eps'] == 1e-5 and parameters['rope_theta'] == 500000.0
    assert parameters['q_scaling_constant'] == 43.7840518911
    authoritative = [config_path, model_root / 'models/muse_spark.py', model_root / 'layers/muse_spark_layers.py',
        FRAMEWORK / 'sglang/kernels/ops/attention/fused_qknorm_rope.py',
        FRAMEWORK / 'sglang/srt/model_executor/forward_batch_info.py',
        FRAMEWORK / 'sglang/srt/managers/schedule_batch.py',
        FRAMEWORK / 'sglang/srt/managers/scheduler_components/batch_result_processor.py',
        derived_root / 'buckets.json', derived_root / 'profile-summary.jsonl',
        ARTIFACT / 'normal-v4-accuracy-only-v1/driver.py', ARTIFACT / 'source_accuracy_v1.py']
    cases = [{'name': 'prefill-wave-3req-12288', 'tokens': 12288, 'wave_requests': 3, 'kind': 'prefill', 'prefix_len': 0},
             {'name': 'prefill-wave-5req-20480', 'tokens': 20480, 'wave_requests': 5, 'kind': 'prefill', 'prefix_len': 0}]
    for output_index in range(1, 16):
        cases.append({'name': f'decode-output-{output_index:02d}-position-{4095 + output_index}',
            'tokens': 8, 'wave_requests': 8, 'kind': 'decode', 'output_index_zero_based': output_index,
            'position': 4095 + output_index})
    result = {'version': VERSION, 'canonical_version': 'canonical-v4-icpx', 'gpu': 6, 'cpus': '32-47',
        'scope': {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'bf16'},
        'parameters': parameters, 'cases': cases, 'source_provider': source,
        'port_site': str(canonical / 'site'), 'canonical_manifest': str(manifest),
        'canonical_manifest_sha256': digest(manifest), 'source_digest': frozen['source_digest'],
        'compiler_policy_sha256': digest(canonical / 'compiler-policy-verified.json'),
        'port_native_SHA': {name: payload['binaries'][name]['sha256'] for name in ('libsgl-ops-sycl-FusedQKNormRope.so', 'libsgl-ops-sycl-FusedQKNormRope_xe35.so')},
        'authoritative_file_SHA': {str(path): digest(path) for path in authoritative},
        'framework_source_package': str(FRAMEWORK), 'model_adapter_normal_import': True,
        'driver_sha256': digest(RUN / 'run_accuracy_r2.py'), 'launcher_sha256': digest(RUN / 'launch_r2.sh'),
        'actual_entry_path': 'Muse forward operator adapter -> sglang fused_qk_norm_rope -> original generic torch.ops.sgl_kernel.fused_qk_norm_rope -> Xe35 guard dispatch for PORT',
        'positions_reconstructed_not_raw_replay': True,
        'raw_trace_available': Path(derived['trace']).is_file(),
        'gap': 'Raw per-request positions/activations/request-order absent; semantic reconstruction assumes fresh equal4096, non-speculative/no-prefix B8 request set. Historical derived trace records 8 decode steps, not all15; remaining OSL16 positions derive scheduler semantics, not raw captured launches.',
        'timing_events_graphs_warmup_allowed': False, 'preopt_rerun': False,
        'all_production_invocations_verified': False}
    for role in ROLES:
        (RUN / role).mkdir()
    (RUN / 'inputs').mkdir()
    save(RUN / 'provenance.json', result)
    save(RUN / 'case-plan.json', cases)


def verify_inputs():
    data = record()
    assert data['version'] == VERSION
    assert digest(RUN / 'run_accuracy_r2.py') == data['driver_sha256']
    assert digest(RUN / 'launch_r2.sh') == data['launcher_sha256']
    for name, checksum in data['authoritative_file_SHA'].items():
        assert digest(local_path(name)) == checksum, f'Frozen provenance/reference changed: {name}'
    assert digest(local_path(data['canonical_manifest'])) == data['canonical_manifest_sha256']
    assert digest(local_path(data['source_provider']['source_manifest'])) == data['source_provider']['source_manifest_sha256']
    return data


def normal_audit(role):
    from runtime_check import audit

    data = verify_inputs()
    site = local_path(data['port_site'] if role == 'port' else data['source_provider']['source_site']).resolve()
    sys.path.insert(0, str(site))
    sys.path.insert(1, str(FRAMEWORK))
    if role == 'port':
        manifest = json.loads(local_path(data['canonical_manifest']).read_text())
        expected = manifest['installed_files'] | manifest['python_snapshot_files']
        provenance = {'canonical_manifest': data['canonical_manifest'], 'canonical_manifest_sha256': data['canonical_manifest_sha256']}
        inventory.verify_inventory(site, expected)
    else:
        provenance = data['source_provider']
        expected = json.loads(local_path(provenance['source_manifest']).read_text())
        assert {str(path.relative_to(site)): digest(path) for path in site.rglob('*') if path.is_file()} == expected
    mapped = audit(provenance, role, site)
    if role == 'port':
        assert len(mapped) == 712 and set(mapped) == {str(site / value['path']) for value in manifest['binaries'].values()}
    import sglang.kernels.ops.attention.fused_qknorm_rope as adapter
    import torch

    assert Path(adapter.__file__).resolve() == FRAMEWORK / 'sglang/kernels/ops/attention/fused_qknorm_rope.py'
    assert adapter.can_use_fused_qk_norm_rope(64, False, torch.bfloat16, device='xpu', rotary_dim=64)
    assert adapter._xpu_op().default._schema.name == 'sgl_kernel::fused_qk_norm_rope'
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if name == 'sglang' or name.startswith('sglang.'):
            if origin:
                assert Path(origin).resolve().is_relative_to(FRAMEWORK)
    return {'mapped_native_SHA': mapped, 'mapped_count': len(mapped), 'site_files_verified': len(expected),
        'adapter_file': str(Path(adapter.__file__).resolve()), 'real_generic_schema': str(adapter._xpu_op().default._schema)}


def invoke(arguments, prefix, environment=None):
    save(prefix.with_suffix('.command.json'), {'argv': arguments, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'PYTHONPATH': (environment or os.environ).get('PYTHONPATH'), 'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK')})
    with prefix.with_suffix('.log').open('w') as output:
        child = subprocess.Popen(arguments, env=environment, stdout=output, stderr=subprocess.STDOUT)
        save(prefix.with_suffix('.pid.json'), {'pid': child.pid, 'ppid': os.getpid()})
        status = child.wait()
    prefix.with_suffix('.exit').write_text(f'{status}\n')
    assert status == 0, f'Actual exit{status}: {prefix}; preserve failure, no autoretry'


def cpu_role(role):
    import torch

    assert not torch.xpu.is_initialized()
    maps = normal_audit(role)
    assert not torch.xpu.is_initialized()
    save(RUN / role / 'cpu-normal.json', {'actual_exit': 0, 'pid': os.getpid(), 'XPU_initialized': False, 'audit': maps})


def cpu_prepare():
    import torch

    assert not torch.xpu.is_initialized() and set(os.sched_getaffinity(0)) == set(range(32, 48))
    data = verify_inputs()
    fixtures = {}
    for index, case in enumerate(data['cases']):
        torch.manual_seed(4200 + index)
        original = torch.randn(case['tokens'], 18432, dtype=torch.bfloat16, device='cpu')
        positions = torch.arange(4096, dtype=torch.int64).repeat(case['wave_requests']) if case['kind'] == 'prefill' else torch.full((8,), case['position'], dtype=torch.int64)
        query_weight = torch.full((64,), data['parameters']['q_scaling_constant'] / 64 ** 0.5, dtype=torch.bfloat16)
        key_weight = torch.ones(64, dtype=torch.bfloat16)
        path = RUN / 'inputs' / f"{case['name']}.pt"
        torch.save({'original': original, 'positions': positions, 'query_weight': query_weight, 'key_weight': key_weight}, path)
        fixtures[case['name']] = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
            'position_min': int(positions.min()), 'position_max': int(positions.max()), 'position_repetitions': case['wave_requests']}
        del original
    save(RUN / 'input-manifest.json', fixtures)
    for role in ROLES:
        invoke([sys.executable, str(RUN / 'run_accuracy_r2.py'), '--mode', 'cpu-role', '--role', role], RUN / role / 'cpu-normal')
    assert not torch.xpu.is_initialized()
    save(RUN / 'cpu-prepare.json', {'actual_exit': 0, 'pid': os.getpid(), 'XPU_initialized': False,
        'fixture_count': len(fixtures), 'normal_source_and_port_preflight_actual_exit': 0})
    print('CPU scope/source/flags/full normal providers/real model adapter/immutable common fixtures PASS; XPU uninitialized', flush=True)


def publish_grant():
    data = verify_inputs()
    assert not GRANT.exists() and (RUN / 'cpu-prepare.exit').read_text().strip() == '0'
    for role in ROLES:
        assert (RUN / role / 'cpu-normal.exit').read_text().strip() == '0'
        assert not json.loads((RUN / role / 'cpu-normal.json').read_text())['XPU_initialized']
    save(GRANT, {'version': VERSION, 'grant_kind': 'model-config-matching-accuracy-only',
        'authorization': 'Explicit main/user independent GPU6 v2 accuracy-only grant; NOT formal performance wave',
        'gpu': 6, 'cpus': '32-47', 'coordinator_release': True, 'start_epoch': time.time() + 45,
        'source_digest': data['source_digest'], 'package_manifest_sha256': data['canonical_manifest_sha256'],
        'compiler_policy_sha256': data['compiler_policy_sha256'],
        'SOURCE_manifest_sha256': data['source_provider']['source_manifest_sha256'],
        'SOURCE_native_sha256': data['source_provider']['source_qk_library_sha256'], 'PORT_native_SHA': data['port_native_SHA'],
        'driver_sha256': data['driver_sha256'], 'launcher_sha256': data['launcher_sha256'],
        'input_manifest_sha256': digest(RUN / 'input-manifest.json'), 'case_plan_sha256': digest(RUN / 'case-plan.json'),
        'scope': data['scope'], 'timing_events_graphs_warmup_allowed': False, 'formal_perf_barrier_consumed': False,
        'preopt_rerun': False, 'W4_lease_transfer_authorized': False})


def verify_grant():
    data = verify_inputs()
    grant = json.loads(GRANT.read_text())
    assert grant['version'] == VERSION and grant['coordinator_release']
    for field, value in {'driver_sha256': data['driver_sha256'], 'launcher_sha256': data['launcher_sha256'],
        'package_manifest_sha256': data['canonical_manifest_sha256'], 'source_digest': data['source_digest'],
        'compiler_policy_sha256': data['compiler_policy_sha256'],
        'SOURCE_manifest_sha256': data['source_provider']['source_manifest_sha256'],
        'SOURCE_native_sha256': data['source_provider']['source_qk_library_sha256'], 'PORT_native_SHA': data['port_native_SHA'],
        'input_manifest_sha256': digest(RUN / 'input-manifest.json'), 'case_plan_sha256': digest(RUN / 'case-plan.json'),
        'scope': data['scope']}.items():
        assert grant[field] == value
    assert not grant['timing_events_graphs_warmup_allowed'] and not grant['formal_perf_barrier_consumed']
    assert not grant['preopt_rerun'] and not grant['W4_lease_transfer_authorized']
    assert os.environ['ZE_AFFINITY_MASK'] == '6' and set(os.sched_getaffinity(0)) == set(range(32, 48))
    return grant


def gpu_role(role):
    verify_grant()
    import torch

    audit_before = normal_audit(role)
    import sglang.kernels.ops.attention.fused_qknorm_rope as adapter

    assert hasattr(torch, 'xpu') and torch.xpu.is_available()
    architecture = tuple(torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device()))
    assert architecture == (3, 5)
    data = record()
    parameters = data['parameters']
    fixtures = json.loads((RUN / 'input-manifest.json').read_text())
    results = {}
    for case in data['cases']:
        fixture = fixtures[case['name']]
        path = local_path(fixture['path'])
        assert digest(path) == fixture['sha256']
        inputs = torch.load(path, map_location='cpu', weights_only=True)
        original = inputs['original']
        qkv = original.to('xpu:0')
        query_weight = inputs['query_weight'].to('xpu:0')
        key_weight = inputs['key_weight'].to('xpu:0')
        positions = inputs['positions'].to('xpu:0')
        assert torch.equal(qkv.cpu().view(torch.uint16), original.view(torch.uint16))
        torch.xpu.synchronize()
        time.sleep(2)
        adapter.fused_qk_norm_rope(qkv, parameters['num_heads_q'], parameters['num_heads_k'], parameters['tail_heads'],
            parameters['head_dim'], parameters['norm_eps'], query_weight, key_weight, parameters['rope_theta'],
            parameters['is_neox'], positions, factor=parameters['factor'], low=parameters['low'], high=parameters['high'],
            attention_factor=parameters['attention_factor'], rotary_dim=parameters['rotary_dim'])
        torch.xpu.synchronize()
        actual = qkv.cpu()
        violations, elements, maximum_error = cpu_reference(original, actual, inputs['positions'].to(torch.int32),
                                                           inputs['query_weight'].float(), inputs['key_weight'].float())
        tail_equal = torch.equal(actual[:, 9216:].view(torch.uint16), original[:, 9216:].view(torch.uint16))
        prefix = RUN / role / case['name']
        torch.save({**inputs, 'actual': actual}, prefix.with_suffix('.pt'))
        result = {'case': case, 'role': role, 'atol': 5e-3, 'rtol': 5e-3, 'violations': violations,
            'elements': elements, 'max_abs': maximum_error, 'tail_bitwise': tail_equal, 'native_invocations': 1,
            'fixture_sha256': fixture['sha256'], 'input_bitwise_checked': True, 'actual_entry': data['actual_entry_path'],
            'positions_original_dtype': 'int64', 'adapter_native_position_dtype': 'int32',
            'timing_events_graphs_warmup_executed': False}
        save(prefix.with_suffix('.json'), result)
        status = 0 if violations == 0 and tail_equal else 1
        prefix.with_suffix('.exit').write_text(f'{status}\n')
        results[case['name']] = result
        save(RUN / role / 'phase-results.json', results)
        print(json.dumps(result), flush=True)
        assert status == 0, f'Strict CPU reference FAIL {case["name"]}; preserve actual exit1, no retry'
        del qkv, original, inputs, actual
    save(RUN / role / 'runtime-audit.json', {'pid': os.getpid(), 'architecture': architecture,
        'before': audit_before, 'after': normal_audit(role), 'case_count': len(results)})


def suite():
    grant = verify_grant()
    assert grant['start_epoch'] > time.time(), 'Fresh independent future epoch required'
    save(RUN / 'gpu-supervisor.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'version': VERSION, 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    for role in ROLES:
        invoke([sys.executable, str(RUN / 'run_accuracy_r2.py'), '--mode', 'gpu-role', '--role', role], RUN / role / 'gpu-role')


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'cpu-prepare', 'cpu-role', 'publish-grant', 'suite', 'gpu-role'), required=True)
    parser.add_argument('--role', choices=ROLES)
    args = parser.parse_args()
    if args.mode in ('cpu-role', 'gpu-role'):
        assert args.role is not None
        (cpu_role if args.mode == 'cpu-role' else gpu_role)(args.role)
    else:
        assert args.role is None
        {'prepare': prepare, 'cpu-prepare': cpu_prepare, 'publish-grant': publish_grant, 'suite': suite}[args.mode]()


if __name__ == '__main__':
    main()

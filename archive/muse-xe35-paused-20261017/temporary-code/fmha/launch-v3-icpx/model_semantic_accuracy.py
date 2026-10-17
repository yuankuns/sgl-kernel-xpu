import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from check_official import canonical_preflight, digest, local_path
from check_preopt import REQUIRED, verify_loaded
from matched_baseline_provenance import package_preflight
from settings import ART, STAGE
from source_only_accuracy import full_cpu_reference, tensor_digest


ROOT = ART / 'model-config51-v1'
FRAMEWORK = local_path('/home/syk/sglang/python/sglang/srt')
BACKEND = FRAMEWORK / 'layers/attention/xpu_backend.py'
WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}


def scope_guard():
    for name, value in {'SERVING_BATCH': '8', 'BATCHES': '8', 'BATCH_SIZE': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        assert os.environ.get(name, value).strip() == value, f'{name} rejected before GPU initialization'


def provider_preflight(provider):
    if provider == 'source':
        package, hashes = package_preflight('source')
        return package, hashes, REQUIRED
    manifest, _, package, required = canonical_preflight()
    return package, {name: record['sha256'] for name, record in manifest['binaries'].items()}, required


def import_framework(torch, cpu_only):
    blocked = []
    original = torch.xpu._lazy_init
    if cpu_only:
        def deny_initialization(*arguments, **keywords):
            blocked.append(traceback.format_stack(limit=12))
            raise RuntimeError('CPU-only preflight forbids XPU initialization; no synthetic architecture result')
        torch.xpu._lazy_init = deny_initialization
    try:
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    finally:
        torch.xpu._lazy_init = original
    if cpu_only:
        assert not torch.xpu.is_initialized()
    return XPUAttentionBackend, RadixAttention, ForwardMode, ForwardContext, forward_context, blocked


def prepare(torch):
    assert not (ROOT / 'plan.json').exists()
    ROOT.mkdir(exist_ok=True)
    generator = torch.Generator(device='cpu').manual_seed(20261017)
    prefill = {'query': torch.randn(32768, 128, 64, dtype=torch.bfloat16, generator=generator),
               'key': torch.randn(512, 64, 16, 64, dtype=torch.bfloat16, generator=generator),
               'value': torch.randn(512, 64, 16, 64, dtype=torch.bfloat16, generator=generator),
               'table': torch.randperm(512, generator=generator).to(torch.int32).reshape(8, 64)}
    decode = {'query': torch.randn(15, 8, 128, 64, dtype=torch.bfloat16, generator=generator),
              'key': torch.randn(520, 64, 16, 64, dtype=torch.bfloat16, generator=generator),
              'value': torch.randn(520, 64, 16, 64, dtype=torch.bfloat16, generator=generator),
              'table': torch.randperm(520, generator=generator).to(torch.int32).reshape(8, 65)}
    bundles = {}
    for name, data in [('prefill-eight-slots.pt', prefill), ('decode-eight-slots.pt', decode)]:
        path = ROOT / name
        torch.save(data, path)
        path.chmod(0o444)
        bundles[name] = {'sha256': digest(path), 'tensors': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'sha256': tensor_digest(value, torch)} for key, value in data.items()}}
    original = json.loads((ART / 'MODEL_CONFIG_MATCHING_CPU_CASE_PLAN_v1.json').read_text())
    cases = []
    for item in original:
        case = dict(item)
        prefill_case = item['name'].startswith('model-prefill')
        case.update({'phase': 'prefill' if prefill_case else 'decode', 'requests': item['active_sequences'],
                     'query_length': 4096 if prefill_case else 1, 'cache_length': item['cache_lengths'][0],
                     'window': item['window_left'], 'window_right': item['window_right_model_call'],
                     'normal_backend_max_seqlen_k_keyword': 'omitted', 'port_wrapper_max_seqlen_k': 0,
                     'source_wrapper_inferred_max_seqlen_k': 4096 if prefill_case else 4160,
                     'port_route': 'genuine upstream generic fallback because max_seqlen_k=0; prefill also excludes active3/5',
                     'bundle': 'prefill-eight-slots.pt' if prefill_case else 'decode-eight-slots.pt',
                     'output_shape': [item['active_sequences'] * (4096 if prefill_case else 1), 128, 64],
                     'prefix_length': 0 if prefill_case else item['cache_lengths'][0] - 1})
        cases.append(case)
    plan = {'version': 'model-config51-v1', 'workload': WORKLOAD, 'cases': cases, 'fixture_files': bundles,
            'harness_sha256': digest(Path(__file__)), 'backend_sha256': digest(BACKEND),
            'model_provenance_sha256': digest(ART / 'MODEL_FIXTURE_PROVENANCE_v1.json'),
            'model_plan_origin_sha256': digest(ART / 'MODEL_CONFIG_MATCHING_CPU_CASE_PLAN_v1.json'),
            'expected_checked_elements_per_provider': 808255488, 'atol': 5e-3, 'rtol': 5e-3,
            'model_entry': 'actual RadixAttention.forward -> actual XPUAttentionBackend.forward -> forward_extend/decode -> normal registered package wrapper/fwd',
            'fixture_boundary': 'synthetic post-attention-projection BF16 tensors; eight serving slots, not activation/rawtrace or actual SWA allocator replay',
            'adapter': 'real RadixAttention and backend methods with preloaded fixture KV/metadata; save_kv_cache=False, no full model weights/inference',
            'full_reference_plan': 'independent CPU FP32 causal/sliding reference; all requests/query/head/channel, chunks64; retain every failure',
            'full_output_plan': 'save every entire BF16 output for source and port; compare input SHA and full byte outputs, report genuine mismatches',
            'no_timing_warmup_events_graphs': True}
    text = json.dumps(plan, indent=2) + '\n'
    (ROOT / 'plan.json').write_text(text)
    (ROOT / 'plan.json').chmod(0o444)
    return plan


def verify_plan():
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert plan['workload'] == WORKLOAD and len(plan['cases']) == 51
    assert sum(case['phase'] == 'prefill' for case in plan['cases']) == 6
    assert plan['harness_sha256'] == digest(Path(__file__)) and plan['backend_sha256'] == digest(BACKEND)
    assert plan['model_provenance_sha256'] == digest(ART / 'MODEL_FIXTURE_PROVENANCE_v1.json')
    for name, record in plan['fixture_files'].items():
        assert digest(ROOT / name) == record['sha256']
    assert sum(case['requests'] * case['query_length'] * 128 * 64 for case in plan['cases']) == 808255488
    tree = ast.parse(BACKEND.read_text())
    backend_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'XPUAttentionBackend')
    for method in ('forward_extend', 'forward_decode'):
        node = next(node for node in backend_class.body if isinstance(node, ast.FunctionDef) and node.name == method)
        calls = [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'flash_attn_with_kvcache']
        assert calls and all('max_seqlen_k' not in {keyword.arg for keyword in call.keywords} for call in calls)
    return plan


def fixture(case, bundle, torch, device):
    if case['phase'] == 'prefill':
        first_slot = case['synthetic_request_slots'][0]
        query = bundle['query'][first_slot * 4096:(first_slot + case['requests']) * 4096]
        table = bundle['table'][first_slot:first_slot + case['requests']]
    else:
        query = bundle['query'][case['cache_length'] - 4097]
        table = bundle['table']
    lengths = torch.full((case['requests'],), case['cache_length'], dtype=torch.int32)
    offsets = torch.arange(case['requests'] + 1, dtype=torch.int32) * case['query_length']
    return tuple(tensor.contiguous().to(device) for tensor in (query, bundle['key'], bundle['value'], table, lengths, offsets))


class FixturePool:
    def __init__(self, key, value):
        self.key, self.value = key, value

    def get_kv_buffer(self, layer_id):
        assert layer_id in (0, 1, 2)
        return self.key, self.value


def backend_call(case, tensors, torch, framework):
    backend_class, layer_class, mode_class, context_class, context_manager, _ = framework
    query, key, value, table, lengths, offsets = tensors
    backend = object.__new__(backend_class)
    backend.use_mla = False
    backend.page_size = 64
    backend.attention_chunk_size = None
    backend.use_sliding_window_kv_pool = True
    backend.kv_cache_dtype_str = 'auto'
    backend.topk = 0
    backend.token_to_kv_pool = FixturePool(key, value)
    backend.forward_metadata = SimpleNamespace(page_table=table, swa_page_table=table, swa_spec_metadata=None,
        cache_seqlens_int32=lengths, cu_seqlens_q=offsets, cu_seqlens_k=lengths,
        max_seq_len_q=case['query_length'], local_attn_metadata=None)
    layer = layer_class(num_heads=128, head_dim=64, scaling=0.125, num_kv_heads=16,
                        layer_id=case['layer'], sliding_window_size=case['window'], logit_cap=0.0)
    mode = mode_class.EXTEND if case['phase'] == 'prefill' else mode_class.DECODE
    batch = SimpleNamespace(forward_mode=mode, spec_info=None, batch_size=case['requests'], _attn_output=None)
    active_keys = key[table].reshape(case['requests'], -1, 16, 64)[:, :case['cache_length']]
    active_values = value[table].reshape(case['requests'], -1, 16, 64)[:, :case['cache_length']]
    if case['phase'] == 'decode':
        active_keys, active_values = active_keys[:, -1:], active_values[:, -1:]
    with context_manager(context_class(attn_backend=backend)):
        return layer(query, active_keys, active_values, batch, save_kv_cache=False).reshape(-1, 128, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=('source', 'port'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    scope_guard()
    package, hashes, required = provider_preflight(args.provider)
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    assert not torch.xpu.is_initialized()
    if args.provider == 'source':
        prior_import = json.loads((ART / 'matched-native-icpx-v1/source/normal-import-verified.json').read_text())
        native_names = {Path(name).name for name in prior_import['loaded_normal_package_libraries_sha256']}
    else:
        native_names = {name for name in hashes if name.endswith('.so')}
    verify_loaded(package, hashes, native_names | required)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    if args.prepare:
        assert args.cpu_only and args.provider == 'source'
        prepare(torch)
    plan = verify_plan()
    if not args.cpu_only:
        assert args.grant is not None and os.environ.get('ZE_AFFINITY_MASK') == '5'
        assert os.sched_getaffinity(0) == set(range(16, 32))
        grant = json.loads(args.grant.read_text())
        assert grant['coordinator_release'] is True and grant['gpu'] == 5 and grant['workload'] == WORKLOAD
        assert grant['phase'] == 'model-config51-normal-accuracy-only' and grant['official_performance_wave'] is False
        assert grant['plan_sha256'] == digest(ROOT / 'plan.json')
        assert grant['canonical_manifest_sha256'] == digest(STAGE / 'source-binary-manifest.json')
        for provider in ('source', 'port'):
            assert grant[f'{provider}_cpu_gate_sha256'] == digest(ROOT / f'{provider}-CPU-gate.json')
            assert (ART / f'model51-cpu-{provider}-v2.exit').read_text().strip() == '0'
        while time.time() < grant['epoch']:
            time.sleep(0.25)
    framework = import_framework(torch, args.cpu_only)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    bundles = {name: torch.load(ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
    if args.cpu_only:
        records = []
        for case in plan['cases']:
            tensors = fixture(case, bundles[case['bundle']], torch, 'cpu')
            query, key, value, table, lengths, offsets = tensors
            assert list(query.shape) == case['output_shape'] and query.dtype == key.dtype == value.dtype == torch.bfloat16
            assert table.dtype == lengths.dtype == offsets.dtype == torch.int32
            assert table.min().item() >= 0 and table.max().item() < key.shape[0]
            assert lengths.tolist() == case['cache_lengths'] and offsets.tolist() == case['cu_seqlens_q']
            assert case['serving_batch'] == 8 and case['window'] in (-1, 2047)
            inspect.signature(flash_attn_with_kvcache).bind(q=query, k_cache=key, v_cache=value, page_table=table,
                cache_seqlens=lengths, cu_seqlens_q=offsets, cu_seqlens_k_new=None, max_seqlen_q=case['query_length'],
                softmax_scale=0.125, causal=True, window_size=(case['window'], case['window_right']), softcap=0.0,
                k_descale=None, v_descale=None, return_softmax_lse=False)
            records.append({'name': case['name'], 'input_sha256': [tensor_digest(tensor, torch) for tensor in tensors]})
        assert not torch.xpu.is_initialized()
        verify_loaded(package, hashes, native_names | required)
        report = {'provider': args.provider, 'CPU_gate': True, 'plan_sha256': digest(ROOT / 'plan.json'),
                  'normal_package': str(package), 'mapped_native_count': len(native_names), 'native_sha256': {name: hashes[name] for name in sorted(native_names)},
                  'schema': str(torch.ops.sgl_kernel.fwd.default._schema),
                  'GPU_initialized': False, 'backend_import_fail_closed_init_attempts': framework[-1],
                  'real_backend_and_registered_wrapper_identity': True, 'cases': records,
                  'full_reference_and_output_plan_confirmed': True, 'no_timing_warmup_events_graphs': True}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'provider': args.provider, 'CPU_gate': True, 'cases': len(records), 'mapped_native_count': len(native_names),
                          'blocked_import_init_attempts': len(framework[-1]), 'GPU_initialized': False}), flush=True)
        return
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    result = {'provider': args.provider, 'runtime_container_pid': os.getpid(), 'grant_sha256': digest(args.grant),
              'plan_sha256': digest(ROOT / 'plan.json'), 'workload': WORKLOAD, 'normal_package': str(package),
              'mapped_native_count': len(native_names), 'backend_sha256': digest(BACKEND),
              'source_digest': grant['source_digest'], 'canonical_manifest_sha256': grant['canonical_manifest_sha256'],
              'no_timing_warmup_events_graphs': True, 'cases': []}
    output_dir = ROOT / f'{args.provider}-full-outputs'
    output_dir.mkdir(exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'runtime_container_pid': os.getpid(), 'provider': args.provider}), flush=True)
    with torch.inference_mode():
        for case in plan['cases']:
            torch.xpu.synchronize()
            time.sleep(2)
            print(json.dumps({'starting_case': case['name'], 'provider': args.provider, 'port_route': case['port_route']}), flush=True)
            tensors = fixture(case, bundles[case['bundle']], torch, 'xpu')
            record = {'name': case['name'], 'input_sha256': [tensor_digest(tensor, torch) for tensor in tensors],
                      'port_route': case['port_route'], 'requests': case['requests'], 'serving_batch': 8, 'native_calls': 1}
            result['cases'].append(record)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            try:
                output = backend_call(case, tensors, torch, framework)
                torch.xpu.synchronize()
                actual = output.cpu()
                record['output_sha256'] = tensor_digest(actual, torch)
                output_file = output_dir / f'{case["name"]}.pt'
                torch.save(actual, output_file)
                record['saved_full_output_file_sha256'] = digest(output_file)
                record.update(full_cpu_reference(SimpleNamespace(**case), (*tensors, output), torch))
                if args.provider == 'port':
                    source_raw = json.loads((ROOT / 'source-accuracy.json').read_text())
                    source_record = next(item for item in source_raw['cases'] if item['name'] == case['name'])
                    assert record['input_sha256'] == source_record['input_sha256']
                    source_output = torch.load(ROOT / 'source-full-outputs' / f'{case["name"]}.pt', map_location='cpu', weights_only=True)
                    assert tensor_digest(source_output, torch) == source_record['output_sha256']
                    record['full_output_bitwise'] = torch.equal(actual.view(torch.uint8), source_output.view(torch.uint8))
                    record['source_port_max_abs'] = (actual.float() - source_output.float()).abs().max().item()
                record['accuracy'] = 'PASS' if record['reference_PASS'] else 'FAIL'
            except Exception as error:
                record['accuracy'] = 'FAIL'
                record['failure'] = repr(error)
                record['traceback'] = traceback.format_exc()
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(record), flush=True)
                raise
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            torch.xpu.synchronize()
            del tensors, output, actual
    verify_loaded(package, hashes, native_names | required)
    result['all51_full_reference_PASS'] = len(result['cases']) == 51 and all(case['reference_PASS'] for case in result['cases'])
    result['actual_mapped_native_sha256'] = {name: hashes[name] for name in sorted(native_names)}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all51_full_reference_PASS']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

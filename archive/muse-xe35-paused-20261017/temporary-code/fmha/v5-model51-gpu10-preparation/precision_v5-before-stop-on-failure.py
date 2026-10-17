import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
VERSION = 'canonical-v5-icpx'
EXPECTED_DIGEST = 'bce898956487b009425edb479cf26ba7e926a67572fc57943d836a8b0f0931be'
WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}
PHASE = 'fmha-v5-model51-and-regression38-gpu10-precision-only'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def scope_guard():
    for name, value in {'SERVING_BATCH': '8', 'BATCHES': '8', 'BATCH_SIZE': '8',
                        'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, value).strip() != value:
            raise RuntimeError(f'{name} rejected before torch or GPU initialization')
    if os.environ.get('FMHA_CANONICAL_VERSION', VERSION) != VERSION:
        raise RuntimeError('only fresh canonical-v5-icpx package is permitted')


def validate_grant(grant_path, now=None):
    if grant_path is None or not grant_path.is_file():
        raise RuntimeError('fresh main GPU10 precision-only grant required; no issuance in this launcher')
    grant = json.loads(grant_path.read_text())
    assert grant['phase'] == PHASE and grant['gpu'] == 10 and grant['runtime_cpus'] == '16-31'
    assert grant['canonical_version'] == VERSION and grant['source_digest'] == EXPECTED_DIGEST
    assert grant['workload'] == WORKLOAD and grant['coordinator_release'] is True
    assert grant['no_timing_warmup_events_graphs'] is True
    assert grant['official_performance_wave'] is False and grant['NV_handoff_authorized'] is False
    assert grant['FM_GPU5_performance_lease_retained'] is True
    assert grant['epoch'] >= grant['issued_epoch'] + 120
    assert (time.time() if now is None else now) >= grant['epoch']
    assert grant['runtime_reuse_permitted'] is False and grant['closed'] is False
    assert grant['harness_sha256'] == digest(Path(__file__))
    assert grant['plan_sha256'] == digest(ART / 'model-config51-v1/plan.json')
    assert grant['source51_raw_sha256'] == digest(ART / 'model-config51-v1/source-accuracy.json')
    assert grant['source38_raw_sha256'] == digest(ART / 'source-only-v1-fresh-optimized.json')
    return grant


def shared_imports():
    os.environ['FMHA_CANONICAL_VERSION'] = VERSION
    sys.path.insert(0, str(ART / 'launch-v3-icpx'))
    sys.path.insert(0, str(ART / 'v5-metadata-bridge-preparation'))
    import model_semantic_accuracy as original
    import prepare_and_check as guards
    from check_official import canonical_preflight, local_path
    from check_preopt import verify_loaded
    from source_only_accuracy import full_cpu_reference, tensor_digest
    return original, guards, canonical_preflight, local_path, verify_loaded, full_cpu_reference, tensor_digest


def mapped_names(package, hashes):
    names = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        value = line.split()[-1]
        if 'common_ops.' not in value and 'libsgl-ops-' not in value and 'libsgl_jit' not in value:
            continue
        path = Path(value).resolve()
        assert path.parent == package.resolve() and digest(path) == hashes[path.name]
        names.add(path.name)
    assert names == set(hashes), (set(hashes) - names, names - set(hashes))
    return names


def verify_native_binding(package):
    libraries = {role: package / f'libsgl-ops-sycl-{role}.so' for role in (
        'entry', 'flash_attention', 'fmha_prefill_64_bf16', 'fmha_decode_64_bf16')}
    symbols = {role: subprocess.check_output(['nm', '-D', '-C', str(path)], text=True)
               for role, path in libraries.items()}
    assert ' T mha_fwd(' in symbols['entry'] and ' U mha_fwd_xe35(' in symbols['entry']
    assert ' T mha_fwd_xe35(' in symbols['flash_attention']
    device_images = {}
    for phase in ('prefill', 'decode'):
        target = f'fmha_{phase}_64_bf16'
        symbol = f'dispatch_xe35_{phase}_64_bf16_xe35('
        assert ' U ' + symbol in symbols['flash_attention'] and ' T ' + symbol in symbols[target]
        sections = subprocess.check_output(['readelf', '-SW', str(libraries[target])], text=True)
        assert '__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen' in sections
        device_images[target] = [line.strip() for line in sections.splitlines() if 'sycl-spir64_gen' in line]
    entry_raw = subprocess.check_output(['nm', '-D', str(libraries['entry'])], text=True)
    public_symbol = next(line.split()[-1] for line in entry_raw.splitlines() if ' T _Z7mha_fwd' in line)
    common = next(package.glob('common_ops.*.so'))
    common_raw = subprocess.check_output(['nm', '-D', str(common)], text=True)
    assert any(line.endswith(public_symbol) and ' U ' in line for line in common_raw.splitlines())
    return {'common_ops_to_entry_exact_ABI_symbol': public_symbol,
            'private_root_and_64BF16_dispatch_symbols': symbols, 'device_image_sections': device_images,
            'boundary': 'static ELF plus genuine mapped SHA and observed fwd metadata; not device kernel profiling'}


def native_route_observer(torch, guards, entry_text, rows):
    from torch.utils._python_dispatch import TorchDispatchMode

    def tensor_metadata(tensor):
        if tensor is None:
            return guards.MetadataTensor([], present=False)
        dtype = 'BF16' if tensor.dtype == torch.bfloat16 else str(tensor.dtype)
        return guards.MetadataTensor(list(tensor.shape), dtype=dtype, contiguous=tensor.is_contiguous())

    class ObserveOriginalFwd(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            keywords = {} if kwargs is None else kwargs
            if function == torch.ops.sgl_kernel.fwd.default:
                arguments = {item.name: item.default_value for item in function._schema.arguments}
                arguments.update(dict(zip((item.name for item in function._schema.arguments), args)))
                arguments.update(keywords)
                context = {name: arguments[name] for name in ('max_seqlen_q', 'max_seqlen_k',
                    'is_causal', 'window_size_left', 'softcap', 'num_kv_splits', 'sm_margin')}
                for name in ('q', 'k', 'v', 'out', 'page_table', 'softmax_lse'):
                    context[name] = tensor_metadata(arguments[name])
                for name in ('q_v', 'kv_batch_idx', 'leftpad_k', 'rotary_cos', 'rotary_sin',
                             'seqlens_rotary', 'scheduler_metadata', 'rel_bias',
                             'bidirectional_block_ids', 'sinks', 'q_descale', 'k_descale', 'v_descale'):
                    context[name + '_'] = tensor_metadata(arguments[name])
                offsets = arguments['cu_seqlens_q']
                context['batch_xe35'] = offsets.numel() - 1 if offsets.dim() == 1 else 0
                context['INT_MAX'] = guards.INT_MAX
                route = guards.evaluate(entry_text, context, True)
                row = {'operator': str(function), 'actual_fwd_maxK_before_private_resolver': arguments['max_seqlen_k'],
                    'active_sequences': context['batch_xe35'], 'Q_shape': list(arguments['q'].shape),
                    'K_shape': list(arguments['k'].shape),
                    'page_table_shape': list(arguments['page_table'].shape) if arguments['page_table'] is not None else None,
                    'causal': arguments['is_causal'], 'window_left': arguments['window_size_left'],
                    'window_right': arguments['window_size_right'], 'guard_evaluation_from_actual_metadata': route,
                    'native_operator_completed': False,
                    'observation_boundary': 'unmodified original fwd redispatch plus exact installed entry SHA, deterministic guards, ELF/native map binding; no device-event kernel trace'}
                rows.append(row)
                returned = function(*args, **keywords)
                row['native_operator_completed'] = True
                return returned
            return function(*args, **keywords)

    return ObserveOriginalFwd()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    scope_guard()
    grant = None
    if not args.cpu_only:
        grant = validate_grant(args.grant)
    if args.guard_only:
        _, _, canonical_preflight, _, _, _, _ = shared_imports()
        manifest, manifest_path, _, _ = canonical_preflight()
        assert manifest['source']['source_digest'] == EXPECTED_DIGEST
        assert grant['canonical_manifest_sha256'] == digest(manifest_path)
        assert grant['compiler_policy_sha256'] == digest(ART / VERSION / 'compiler-policy-verified.json')
        audit_path = ART / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json'
        assert grant['coordinator_full_audit_sha256'] == digest(audit_path)
        audit = json.loads(audit_path.read_text())
        assert audit['actual_exit'] == 0 and audit['source_digest'] == EXPECTED_DIGEST
        cpu_gate = ROOT / 'normal-v5-CPU-gate.json'
        assert grant['normal_CPU_gate_sha256'] == digest(cpu_gate)
        gate = json.loads(cpu_gate.read_text())
        assert gate['CPU_gate'] and not gate['GPU_initialized'] and len(gate['cases']) == 51
        assert len(gate['regression38_case_names']) == 38
        assert gate['canonical_manifest_sha256'] == digest(manifest_path)
        assert gate['mapped_native_sha256'] == {name: record['sha256'] for name, record in manifest['binaries'].items()}
        assert gate['harness_sha256'] == digest(Path(__file__))
        print('fresh GPU10 precision grant, full/install/post/audit0 and normal CPU gate checked; no torch import')
        return
    assert args.output is not None and not args.output.exists()
    original, guards, canonical_preflight, local_path, verify_loaded, full_cpu_reference, tensor_digest = shared_imports()
    manifest, manifest_path, package, required = canonical_preflight()
    assert manifest['source']['source_digest'] == EXPECTED_DIGEST
    stage = ART / VERSION
    audit_path = ART / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json'
    audit = json.loads(audit_path.read_text())
    assert audit['actual_exit'] == audit['default_all_actual_exit'] == audit['install_actual_exit'] == 0
    assert audit['source_digest'] == EXPECTED_DIGEST and audit['package_manifest_sha256'] == digest(manifest_path)
    assert audit['only_source_delta'] == ['src/sycl/xe35/fmha/entry.cpp']
    hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
    binding = verify_native_binding(package)
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    entry = local_path(candidate['destination']) / 'src/sycl/xe35/fmha/entry.cpp'
    assert digest(entry) == manifest['source']['files']['src/sycl/xe35/fmha/entry.cpp']['sha256']
    entry_text = entry.read_text()
    plan = original.verify_plan()
    source51_path = ART / 'model-config51-v1/source-accuracy.json'
    source51 = json.loads(source51_path.read_text())
    source38_path = ART / 'source-only-v1-fresh-optimized.json'
    source38 = json.loads(source38_path.read_text())
    assert source51['all51_full_reference_PASS'] and source38['all_38_reference_PASS']
    assert (ART / 'model51-accuracy-source-v1.exit').read_text().strip() == '0'
    assert (ART / 'source-only-v1-fresh-optimized.exit').read_text().strip() == '0'
    from matched_baseline_provenance import package_preflight
    package_preflight('source')
    source_binding = json.loads((ART / 'MODEL51_SOURCE_ROUTE_RUNTIME_BINDING_CPU_v1.json').read_text())
    assert source_binding['runtime_actual_mapped_3_tested_DSO_matches_CPU_route_proof']
    assert source_binding['source_optimized_prefill'] == 6 and source_binding['source_optimized_decode'] == 45
    assert source_binding['runtime_raw_sha256'] == digest(source51_path)
    source51_records = {record['name']: record for record in source51['cases']}
    source38_records = {record['name']: record for record in source38['cases']}
    for name, record in source51_records.items():
        assert digest(ART / 'model-config51-v1/source-full-outputs' / f'{name}.pt') == record['saved_full_output_file_sha256']
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve() and not torch.xpu.is_initialized()
    verify_loaded(package, hashes, set(hashes))
    names = mapped_names(package, hashes)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    if grant is not None:
        assert os.environ.get('ZE_AFFINITY_MASK') == '10' and os.sched_getaffinity(0) == set(range(16, 32))
        assert grant['canonical_manifest_sha256'] == digest(manifest_path)
        assert grant['compiler_policy_sha256'] == digest(stage / 'compiler-policy-verified.json')
        assert grant['coordinator_full_audit_sha256'] == digest(audit_path)
        cpu_gate = ROOT / 'normal-v5-CPU-gate.json'
        assert grant['normal_CPU_gate_sha256'] == digest(cpu_gate)
        gate = json.loads(cpu_gate.read_text())
        assert gate['CPU_gate'] and not gate['GPU_initialized'] and gate['canonical_manifest_sha256'] == digest(manifest_path)
    framework = original.import_framework(torch, args.cpu_only)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    bench = runpy.run_path(str(local_path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35/benchmark/bench_muse_fmha.py')))
    _, regression = bench['arguments_xe35'](['--guard-only', '--mode', 'accuracy', '--reference', 'full'])
    assert len(regression) == 38
    bundles = {name: torch.load(original.ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
    result = {'canonical_version': VERSION, 'source_digest': EXPECTED_DIGEST, 'CPU_gate': args.cpu_only,
        'GPU_initialized': torch.xpu.is_initialized(), 'gpu': None if args.cpu_only else 10,
        'runtime_container_pid': os.getpid(), 'workload': WORKLOAD,
        'canonical_manifest_sha256': digest(manifest_path), 'harness_sha256': digest(Path(__file__)),
        'compiler_policy_sha256': digest(stage / 'compiler-policy-verified.json'),
        'source51_raw_sha256': digest(source51_path), 'source38_raw_sha256': digest(source38_path),
        'mapped_native_sha256': {name: hashes[name] for name in sorted(names)},
        'normal_native_ELF_binding_proof': binding,
        'grant_sha256': None if args.cpu_only else digest(args.grant),
        'no_timing_warmup_events_graphs': True, 'cases': []}
    if args.cpu_only:
        for case in plan['cases']:
            tensors = original.fixture(case, bundles[case['bundle']], torch, 'cpu')
            query, key, value, table, lengths, offsets = tensors
            input_hashes = [tensor_digest(tensor, torch) for tensor in tensors]
            assert input_hashes == source51_records[case['name']]['input_sha256']
            inspect.signature(flash_attn_with_kvcache).bind(q=query, k_cache=key, v_cache=value,
                page_table=table, cache_seqlens=lengths, cu_seqlens_q=offsets,
                max_seqlen_q=case['query_length'], softmax_scale=0.125, causal=True,
                window_size=(case['window'], case['window_right']), softcap=0.0, return_softmax_lse=False)
            result['cases'].append({'name': case['name'], 'input_sha256': input_hashes})
        assert not torch.xpu.is_initialized()
        result['regression38_case_names'] = [case.name for case in regression]
        result['real_backend_and_registered_wrapper_identity'] = True
        result['backend_import_fail_closed_init_attempts'] = framework[-1]
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        return
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    output_dir = ROOT / 'normal-v5-full-outputs'
    output_dir.mkdir(exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    work = [('model51', SimpleNamespace(**case), case) for case in plan['cases']]
    work += [('regression38', case, None) for case in regression]
    with torch.inference_mode():
        for family, case, model_case in work:
            torch.xpu.synchronize()
            time.sleep(2)
            record = {'family': family, 'name': case.name, 'serving_batch': 8, 'registered_fwd_calls': 0}
            result['cases'].append(record)
            print(json.dumps({'starting_case': case.name, 'family': family, 'gpu': 10}), flush=True)
            try:
                routes = []
                if family == 'model51':
                    tensors = original.fixture(model_case, bundles[model_case['bundle']], torch, 'xpu')
                    input_hashes = [tensor_digest(tensor, torch) for tensor in tensors]
                    baseline = source51_records[case.name]
                    assert input_hashes == baseline['input_sha256']
                    with native_route_observer(torch, guards, entry_text, routes):
                        output = original.backend_call(model_case, tensors, torch, framework)
                    source_file = ART / 'model-config51-v1/source-full-outputs' / f'{case.name}.pt'
                else:
                    tensors_with_output = bench['tensors_xe35'](case, torch)
                    tensors, output = tensors_with_output[:-1], tensors_with_output[-1]
                    input_hashes = [tensor_digest(tensor, torch) for tensor in tensors]
                    baseline = source38_records[case.name]
                    assert input_hashes == baseline['input_sha256']
                    with native_route_observer(torch, guards, entry_text, routes):
                        bench['launch_xe35'](case, tensors_with_output, flash_attn_with_kvcache)
                    source_file = ART / 'matched-native-icpx-v1/source-only-paired-v1' / f'{case.name}.pt'
                record['actual_fwd_observations'] = routes
                record['registered_fwd_calls'] = len(routes)
                observed_once = len(routes) == 1 and routes[0]['native_operator_completed']
                route = routes[0]['guard_evaluation_from_actual_metadata'] if observed_once else {'optimized': False}
                route_ok = observed_once and route['optimized']
                if family == 'model51' and route_ok:
                    route_ok = routes[0]['actual_fwd_maxK_before_private_resolver'] == 0 and route['private_maxK'] == model_case['source_wrapper_inferred_max_seqlen_k']
                record['actual_metadata_private_route_PASS'] = route_ok
                device_native = f'libsgl-ops-sycl-fmha_{"decode" if case.query_length == 1 else "prefill"}_64_bf16.so'
                assert device_native in names and 'libsgl-ops-sycl-entry.so' in names
                if route_ok:
                    record['deterministic_private_device_native'] = device_native
                    record['device_native_sha256'] = hashes[device_native]
                record['root_entry_sha256'] = hashes['libsgl-ops-sycl-entry.so']
                torch.xpu.synchronize()
                actual = output.cpu().contiguous()
                output_file = output_dir / f'{family}-{case.name}.pt'
                torch.save(actual, output_file)
                record['saved_full_output_file_sha256'] = digest(output_file)
                record['input_sha256'] = input_hashes
                record['output_sha256'] = tensor_digest(actual, torch)
                record.update(full_cpu_reference(case, (*tensors, output), torch))
                source_output = torch.load(source_file, map_location='cpu', weights_only=True).contiguous()
                assert tensor_digest(source_output, torch) == baseline['output_sha256']
                assert actual.shape == source_output.shape and actual.dtype == source_output.dtype == torch.bfloat16
                record['full_output_bitwise'] = torch.equal(actual.view(torch.uint8), source_output.view(torch.uint8))
                record['source_port_max_abs'] = (actual.float() - source_output.float()).abs().max().item()
                record['accuracy'] = 'PASS' if route_ok and record['reference_PASS'] and record['full_output_bitwise'] else 'FAIL'
            except Exception as error:
                record.update({'accuracy': 'FAIL', 'failure': repr(error), 'traceback': traceback.format_exc()})
                record.setdefault('actual_fwd_observations', routes)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            torch.xpu.synchronize()
    verify_loaded(package, hashes, set(hashes))
    result['all89_fullrefs_and_source_wholebitwise_PASS'] = len(result['cases']) == 89 and all(record['accuracy'] == 'PASS' for record in result['cases'])
    result['GPU_initialized'] = torch.xpu.is_initialized()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all89_fullrefs_and_source_wholebitwise_PASS']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, AssertionError, FileNotFoundError) as error:
        print(f'precision gate rejected: {error}', file=sys.stderr)
        raise SystemExit(2)

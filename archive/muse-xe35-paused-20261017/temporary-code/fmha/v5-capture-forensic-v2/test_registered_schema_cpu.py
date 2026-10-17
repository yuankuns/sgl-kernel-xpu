import argparse
import json
from pathlib import Path

import forensic_runner as forensic


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'port'), required=True)
    args = parser.parse_args()
    forensic.precision.scope_guard()
    import torch
    import sgl_kernel
    from torch.utils._python_dispatch import TorchDispatchMode

    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    old_gate_path = ART / 'v5-capture-forensic-v1' / f'{args.role}-CPU-gate.json'
    old_gate = json.loads(old_gate_path.read_text())
    package = Path(old_gate['normal_package'])
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    expected = old_gate['mapped_native_SHA']
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = Path(line.split()[-1])
        if path.name in expected:
            assert path.resolve().parent == package.resolve()
            mapped.add(path.name)
    assert mapped == set(expected)
    targets = ('libsgl-ops-sycl-entry.so', 'libsgl-ops-sycl-upstream_adapter.so') if args.role == 'port' else (
        'libsgl-ops-sycl-flash_attentionXe35.so',
        'libsgl-ops-sycl-flash_attentionXe35_decode_64_bf16.so',
        'libsgl-ops-sycl-flash_attentionXe35_prefill_64_bf16.so')
    for name in targets:
        assert forensic.digest(package / name) == expected[name]
    operator = torch.ops.sgl_kernel.fwd.default
    schema = operator._schema
    names = [item.name for item in schema.arguments]
    assert 'is_causal' in names and 'causal' not in names
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    original, *_ = forensic.precision.shared_imports()
    framework = original.import_framework(torch, True)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    plan = original.verify_plan()
    case = plan['cases'][0]
    assert case['name'] == forensic.CASE_NAME and case['serving_batch'] == 8
    bundle = torch.load(original.ROOT / case['bundle'], map_location='cpu', weights_only=True)
    tensors = original.fixture(case, bundle, torch, 'cpu')
    assert [forensic.tensor_metadata(tensor)['shape'] for tensor in tensors][:4] == [
        [12288, 128, 64], [512, 64, 16, 64], [512, 64, 16, 64], [3, 64]]
    captured = []

    class StopBeforeNative(Exception):
        pass

    class ObserveRegisteredCPUArguments(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if function == operator:
                keywords = {} if kwargs is None else kwargs
                arguments = forensic.parse_fwd_arguments(schema, role, torch.bfloat16, args, keywords)
                captured.append((args, keywords, arguments))
                raise StopBeforeNative('actual registered fwd intercepted before any CPU native implementation')
            return function(*args, **({} if kwargs is None else kwargs))

    def deny_xpu(*arguments, **keywords):
        raise AssertionError('CPU schema contract forbids XPU initialization')

    role = args.role
    lazy_init = torch.xpu._lazy_init
    torch.xpu._lazy_init = deny_xpu
    try:
        try:
            with ObserveRegisteredCPUArguments():
                original.backend_call(case, tensors, torch, framework)
        except StopBeforeNative:
            pass
        assert len(captured) == 1
        positional, keywords, arguments = captured[0]
        assert set(arguments) == set(names)
        assert arguments['max_seqlen_k'] == (4096 if role == 'source' else 0)
        assert arguments['window_size_left'] == 2047 and arguments['window_size_right'] == 0
        assert arguments['softmax_scale'] == 0.125 and arguments['softcap'] == 0.0
        assert arguments['is_causal'] is True
        assert all(value.device.type == 'cpu' for value in arguments.values() if isinstance(value, torch.Tensor))
        tensors_metadata = {name: forensic.tensor_metadata(value) for name, value in arguments.items() if isinstance(value, torch.Tensor)}
        scalars_metadata = {name: value for name, value in arguments.items() if value is None or isinstance(value, (int, float, bool, str))}
        assert set(tensors_metadata) | set(scalars_metadata) == set(names)
        by_keyword = forensic.parse_fwd_arguments(schema, role, torch.bfloat16, kwargs=arguments)
        by_position = forensic.parse_fwd_arguments(schema, role, torch.bfloat16, args=tuple(arguments[name] for name in names))
        assert all(by_keyword[name] is arguments[name] and by_position[name] is arguments[name] for name in names)
        rejected = []
        invalid = [('is_causal', False), ('max_seqlen_q', 1), ('q', arguments['q'][:1]),
            ('q', arguments['q'].view(torch.uint16)), ('page_table', arguments['page_table'][:1])]
        if role == 'source':
            invalid.append(('max_seqlen_k', 0))
        for name, value in invalid:
            changed = dict(arguments, **{name: value})
            try:
                forensic.parse_fwd_arguments(schema, role, torch.bfloat16, kwargs=changed)
            except AssertionError:
                rejected.append(name)
            else:
                raise AssertionError(f'expected original observer guard rejection for {name}')
        assert len(rejected) == len(invalid)
        assert not torch.xpu.is_initialized()
    finally:
        torch.xpu._lazy_init = lazy_init
    source_schema = json.loads((ART / 'v5-capture-forensic-v1/future-run-v1/source/protocol-markers.jsonl').read_text().splitlines()[0])['actual_registered_schema']
    if role == 'source':
        assert str(schema) == source_schema
    report = {
        'actual_exit': 0, 'role': role, 'real_registered_schema': str(schema), 'schema_argument_count': len(names),
        'schema_argument_names': names, 'actual_CPU_backend_registered_call_captured': True,
        'CPU_native_fwd_executed': False, 'GPU_initialized': False, 'native_capture_executed': False,
        'all_observer_read_fields_validated': True, 'positional_keyword_default_mapping_PASS': True,
        'negative_original_guards_rejected': rejected, 'captured_positional_count': len(positional),
        'captured_keyword_names': list(keywords), 'tensors': tensors_metadata, 'scalars': scalars_metadata,
        'normal_package': str(package), 'normal_mapped_count': len(mapped),
        'fresh_target_SHA256': {name: expected[name] for name in targets},
        'reused_full_mapping_hash_provenance': str(old_gate_path),
        'full_package_content_rehash_this_test': False,
        'v1_saved_source_schema_matches_current': str(schema) == source_schema,
        'runner_SHA256': forensic.digest(ROOT / 'forensic_runner.py'),
    }
    with (ROOT / f'{role}-registered-schema-CPU.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    gate = dict(old_gate, runner_sha256=report['runner_SHA256'],
        registered_schema_contract_SHA256=forensic.digest(ROOT / f'{role}-registered-schema-CPU.json'),
        prior_full_hash_CPU_gate_SHA256=forensic.digest(old_gate_path),
        current_normal_maps_names_verified=True, current_target_hashes_verified=True,
        full_package_content_rehash_this_gate=False)
    with (ROOT / f'{role}-CPU-gate.json').open('x') as stream:
        stream.write(json.dumps(gate, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('actual_exit', 'role', 'schema_argument_count',
        'normal_mapped_count', 'CPU_native_fwd_executed', 'GPU_initialized', 'all_observer_read_fields_validated')}))


if __name__ == '__main__':
    main()

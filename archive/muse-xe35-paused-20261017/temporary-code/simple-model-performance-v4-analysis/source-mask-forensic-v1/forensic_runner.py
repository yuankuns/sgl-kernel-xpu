import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
CPU_AUDIT = DIRECTORY / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def preflight():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise RuntimeError('B8-only rejected before Torch/GPU: ' + name)
    for name, value in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, value) != value:
            raise RuntimeError('Fixed production scope rejected before Torch/GPU: ' + name)


def native_rows(text):
    pairs = re.findall(r'<TR><TD[^>]*>(.*?)</TD><TD[^>]*><FONT[^>]*>(.*?)</FONT></TD></TR>', text, re.DOTALL)
    return {html.unescape(re.sub('<[^>]*>', '', name)): html.unescape(re.sub('<[^>]*>', '', value)) for name, value in pairs}


def pointers(tensor):
    return {'data_ptr': tensor.data_ptr(), 'storage_data_ptr': tensor.untyped_storage().data_ptr(), 'storage_bytes': tensor.untyped_storage().nbytes(), 'storage_offset': tensor.storage_offset(), 'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype), 'device': str(tensor.device), 'python_tensor_identity': id(tensor)}


def save_actual(torch, output, name, ids, returned, expected, graph_stage):
    torch.xpu.synchronize()
    actual = {'ids': ids.detach().cpu().clone(), 'mapped': returned[0].detach().cpu().clone(), 'invalid': returned[1].detach().cpu().clone(), 'expected_mapped': expected[0].clone(), 'expected_invalid': expected[1].clone()}
    tensor_path = output / (name + '.pt')
    torch.save(actual, tensor_path)
    metadata = {'stage': graph_stage, 'inputs_outputs_saved_before_assert': True, 'PT_sha256': digest(tensor_path), 'actual_kernel_argument_tensor_bindings': {'arg0_input': pointers(ids), 'arg1_mapped': pointers(returned[0]), 'arg2_invalid': pointers(returned[1])}, 'whole_bits': {key: hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for key, value in actual.items()}}
    save(output / (name + '.json'), metadata)
    torch.testing.assert_close(actual['mapped'], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual['invalid'], expected[1], atol=0, rtol=0)
    assert torch.equal(actual['mapped'].reshape(-1).view(torch.uint8), expected[0].reshape(-1).view(torch.uint8))
    assert torch.equal(actual['invalid'].reshape(-1).view(torch.uint8), expected[1].reshape(-1).view(torch.uint8))


def grant(path):
    preflight()
    record = json.loads(path.read_text())
    expected = {'issuer': 'main-coordinator', 'scope': 'SOURCE-mask-singleton-unmeasured-forensic-v1', 'accuracy_authorized': True, 'performance_authorized': False, 'gpu': 7, 'label': 'before', 'context': 'isolated-mask-only', 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'CPU_affinity': list(range(80, 96)), 'runner_sha256': digest(__file__), 'CPU_audit_sha256': digest(CPU_AUDIT)}
    assert all(type(record.get(key)) is type(value) and record[key] == value for key, value in expected.items()), 'Wrong fresh forensic grant'
    assert record['start_epoch'] - record['issued_epoch'] >= 120
    audit = json.loads(CPU_AUDIT.read_text())
    assert audit['actual_exit'] == 0 and not audit['performance_accepted']
    assert record['SOURCE_native_sha256'] == audit['eager_accuracy_v2_scope']['SOURCE_same_native_SHA']
    comparator_path = ROOT / 'logits-vocab/canonical-validation-v4-icpx/source-comparator-manifest.json'
    assert digest(comparator_path) == record['comparator_sha256']
    comparator = json.loads(comparator_path.read_text())
    assert digest(Path(comparator['source_native_manifest'])) == comparator['source_native_manifest_sha256'] == record['source_native_manifest_sha256']
    assert set(os.sched_getaffinity(0)) == set(range(80, 96)) and os.environ.get('ZE_AFFINITY_MASK') == '7'
    return record, comparator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-authorized', action='store_true')
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    preflight()
    if not args.main_authorized or args.grant is None:
        parser.error('CPU proposal only; fresh main unmeasured accuracy grant required')
    record, comparator = grant(args.grant)
    while time.time() < record['start_epoch']:
        time.sleep(min(1, record['start_epoch'] - time.time()))
    import torch
    import sgl_kernel

    assert not torch.xpu.is_initialized()
    assert Path(sys.executable).absolute() == Path('/opt/gfx-deps/venv/bin/python')
    assert Path(torch.__file__).resolve().parent == Path('/opt/gfx-deps/venv/lib/python3.12/site-packages/torch')
    stage = Path(comparator['stage'])
    assert Path(sgl_kernel.__file__).resolve() == stage / 'sgl_kernel/__init__.py'
    assert Path(sgl_kernel.common_ops.__file__).resolve() == stage / comparator['registration']
    actual = {str(path.relative_to(stage)): digest(path) for path in stage.rglob('*') if path.is_file()}
    assert actual == {name: value['sha256'] for name, value in comparator['files'].items()}, 'Current normal SOURCE package changed'
    tested = stage / 'sgl_kernel/libsgl-ops-sycl-VocabMask.so'
    assert digest(tested) == record['SOURCE_native_sha256']
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libsgl-ops-sycl-VocabMask.so' in line}
    assert mapped == {str(tested)}, 'Wrong actual SOURCE native mapping'
    assert sgl_kernel.is_xe3_arch(), 'Actual Xe3.5 required'
    operation = torch.ops.sgl_kernel.vocab_mask_xe35
    output = DIRECTORY / 'fresh-unmeasured-artifact'
    output.mkdir(exist_ok=False)
    ids_cpu = torch.tensor([-1, 0, 1, 202047, 202048, 100, 70000, 150000], dtype=torch.int64)
    invalid_cpu = (ids_cpu < 0) | (ids_cpu >= 202048)
    expected = (ids_cpu * (~invalid_cpu), invalid_cpu)
    torch.save({'ids': ids_cpu, 'expected_mapped': expected[0], 'expected_invalid': expected[1], 'org_start': 0, 'org_end': 202048, 'num_org_padding': 0, 'added_start': 202048, 'added_end': 202048}, output / 'fresh-inputs.pt')
    save(output / 'inputs-seal.json', {'sha256': digest(output / 'fresh-inputs.pt'), 'fresh_input_not_old_failed_output': True, 'context': record['context'], 'embedding_prefix_reproduced': False})
    ids = ids_cpu.to('xpu')
    torch.xpu.synchronize()
    time.sleep(2)
    returned = []
    def launch():
        returned[:] = operation(ids, 0, 202048, 0, 202048, 202048)
        return returned
    eager_mapped, eager_invalid = launch()
    save_actual(torch, output, 'initial-eager', ids, (eager_mapped, eager_invalid), expected, 'eager before capture')
    warm_pointers = []
    for warmup_index in range(5):
        launch()
        warm_pointers.append({'index': warmup_index, 'mapped': pointers(returned[0]), 'invalid': pointers(returned[1])})
    torch.xpu.synchronize()
    save(output / 'warm-pointers.json', warm_pointers)
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    capture_bindings = None
    status = 1
    try:
        with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
            assert torch.xpu.is_current_stream_capturing()
            launch()
            capture_bindings = {'arg0_input': pointers(ids), 'arg1_mapped': pointers(returned[0]), 'arg2_invalid': pointers(returned[1])}
        save(output / 'capture-tensor-pointers.json', capture_bindings)
        dot_path = output / 'single-call.dot'
        graph.debug_dump(str(dot_path))
        text = dot_path.read_text()
        nodes = re.findall(r'^\s*(L[0-9]+_S[0-9]+_C[0-9]+)\s*\[label\s*=', text, re.MULTILINE)
        rows = native_rows(text)
        save(output / 'native-DOT-before-check.json', {'DOT_sha256': digest(dot_path), 'native_fields': rows, 'direct_capture_tensor_bindings': capture_bindings, 'native_pArguments_address': rows.get('pArguments'), 'native_arg_value_addresses_exposed': {name: value for name, value in rows.items() if name.startswith('arg[') and re.fullmatch(r'0x[0-9A-Fa-f]+', value)}, 'tensor_argument_value_addresses_may_not_be_exposed_by_debug_dump': True, 'pArguments_is_not_input_output_data_ptr': True, 'DOT_saved_before_instantiate_replay_and_check': True})
        assert 'VocabMaskKernel' in rows['kernelName']
        assert len(nodes) == 1 and 'zeCommandListAppendLaunchKernelWithArguments' in text
        graph.instantiate()
        save(output / 'instantiate-pointers.json', {'arg0_input': pointers(ids), 'arg1_mapped': pointers(returned[0]), 'arg2_invalid': pointers(returned[1])})
        graph.replay()
        save_actual(torch, output, 'singleton-replay-before-assert', ids, returned, expected, 'single capture -> DOT -> instantiate -> replay; graph still alive')
        status = 0
    except Exception as error:
        save(output / 'actual-failure.json', {'error_type': type(error).__name__, 'error': str(error), 'actual_exit': 1, 'no_timing': True, 'old_failed_values_inferred': False})
        raise
    finally:
        graph.reset()
        torch.xpu.synchronize()
        save(output / 'terminal.json', {'actual_exit': status, 'events_created': False, 'timing_samples': 0, 'capture_logical_calls': 1, 'graph_reset_after_evidence_and_check': True, 'eager_original_pair_kept_alive': True, 'old_run_repaired_or_accepted': False, 'performance_accepted': False})


if __name__ == '__main__':
    main()

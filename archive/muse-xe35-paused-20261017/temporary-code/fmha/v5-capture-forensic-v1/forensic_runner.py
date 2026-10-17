import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
sys.path.insert(0, str(ART / 'v5-model51-gpu10-preparation'))
import precision_v5 as precision


PHASE = 'fmha-v5-source-port-firstcase-two-generations-capture8-forensic-no-timing'
CAPTURE_COUNT = 8
GENERATIONS = 2
CASE_NAME = 'model-prefill-wave1-active3-layer0'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def check_grant(args):
    precision.scope_guard()
    if args.grant is None or not args.grant.is_file():
        raise RuntimeError('new main forensic grant required before Torch/GPU; old performance grant forbidden')
    grant = json.loads(args.grant.read_text())
    assert grant['phase'] == PHASE and grant['gpu'] == args.gpu_id and args.gpu_id in range(8)
    assert grant['runtime_cpus'] == '16-31' and grant['workload'] == precision.WORKLOAD
    assert grant['coordinator_release'] and not grant['closed'] and not grant['runtime_reuse_permitted']
    assert grant['epoch'] >= grant['issued_epoch'] + 120 and time.time() >= grant['epoch']
    assert grant['no_timing_warmup_events'] and not grant['performance_acceptance']
    assert grant['capture_count'] == CAPTURE_COUNT and grant['generations'] == GENERATIONS
    assert grant['case_name'] == CASE_NAME and grant['roles'] == ['source', 'port']
    assert grant['source_digest'] == precision.EXPECTED_DIGEST and grant['canonical_version'] == precision.VERSION
    assert grant['runner_sha256'] == digest(Path(__file__))
    assert grant['SOURCE_golden_raw_sha256'] == digest(ART / 'model-config51-v1/source-accuracy.json')
    assert grant['model_plan_sha256'] == digest(ART / 'model-config51-v1/plan.json')
    assert grant['source_package_manifest_sha256'] == digest(ART / 'matched-native-icpx-v1/source/normal-package-manifest.json')
    assert grant['normal_CPU_gates_sha256'][args.role] == digest(ROOT / f'{args.role}-CPU-gate.json')
    assert (ROOT / f'{args.role}-CPU-gate.exit').read_text().strip() == '0'
    assert grant['NV_handoff_authorized'] is False
    return grant


def queue_state(torch, device=None):
    stream = torch.xpu.current_stream(device=device)
    pointer = stream.sycl_queue
    assert isinstance(pointer, int) and pointer > 0 and stream._as_parameter_.value == pointer
    return {'actual_API': 'torch.xpu.current_stream(device).sycl_queue', 'sycl_queue_address': pointer,
        'sycl_queue_address_hex': hex(pointer), 'stream_object_id': id(stream), 'device': str(stream.device),
        'pointer_kind': 'real PyTorch native SYCL queue address; not a fabricated LevelZero command-list handle',
        'native_helper_contract': 'SOURCE and PORT private helper both call at::xpu::getCurrentXPUStream(t.device().index()).queue(); source/ELF binding audited, no added C++ queue probe'}


def tensor_metadata(tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'device': str(tensor.device),
        'stride': list(tensor.stride()), 'storage_offset': tensor.storage_offset(),
        'data_ptr': tensor.data_ptr(), 'data_ptr_hex': hex(tensor.data_ptr()),
        'storage_data_ptr': tensor.untyped_storage().data_ptr(), 'python_object_id': id(tensor)}


class Journal:
    def __init__(self, path):
        self.stream = path.open('x')
        self.sequence = 0

    def emit(self, stage, **data):
        self.sequence += 1
        record = {'sequence': self.sequence, 'epoch': time.time(), 'container_pid': os.getpid(), 'stage': stage, **data}
        self.stream.write(json.dumps(record) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        print(json.dumps(record), flush=True)


def snapshot_stage(stage, generation, tensors, output, torch, folder, journal, native_binding, validity):
    journal.emit(f'{stage}_save_begin', generation=generation, queue=queue_state(torch), output_validity=validity)
    torch.xpu.synchronize()
    path = folder / f'g{generation}-{stage}.pt'
    assert not path.exists()
    outputs = [] if output is None else output if isinstance(output, list) else [output]
    saved = {'input_tensors': [tensor.detach().cpu().contiguous() for tensor in tensors],
        'outputs': [tensor.detach().cpu().contiguous() for tensor in outputs]}
    metadata = {'inputs': [tensor_metadata(tensor) for tensor in tensors],
        'outputs': [tensor_metadata(tensor) for tensor in outputs]}
    torch.save(saved, path)
    journal.emit(f'{stage}_saved', generation=generation, saved_path=str(path), saved_file_sha256=digest(path),
        live_tensor_addresses=metadata, queue=queue_state(torch), native_binding=native_binding, output_validity=validity)


def capture_observer(torch, role, journal, generation, rows):
    from torch.utils._python_dispatch import TorchDispatchMode

    class ObserveActualNativeCall(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            keywords = {} if kwargs is None else kwargs
            if function != torch.ops.sgl_kernel.fwd.default:
                return function(*args, **keywords)
            arguments = {item.name: item.default_value for item in function._schema.arguments}
            arguments.update(dict(zip((item.name for item in function._schema.arguments), args)))
            arguments.update(keywords)
            query, key, value, table = arguments['q'], arguments['k'], arguments['v'], arguments['page_table']
            assert query.shape == (12288, 128, 64) and key.shape == value.shape == (512, 64, 16, 64)
            assert query.dtype == key.dtype == value.dtype == torch.bfloat16 and table.shape == (3, 64)
            assert arguments['max_seqlen_q'] == 4096 and arguments['causal'] is True
            if role == 'source':
                assert arguments['max_seqlen_k'] == 4096
            tensor_args = {name: tensor_metadata(value) for name, value in arguments.items() if isinstance(value, torch.Tensor)}
            scalar_args = {name: value for name, value in arguments.items() if value is None or isinstance(value, (int, float, bool, str))}
            row = {'call': len(rows) + 1, 'role': role, 'registered_operator': str(function),
                'queue_before': queue_state(torch, arguments['q'].device), 'tensors': tensor_args,
                'scalars': scalar_args, 'native_returned': False,
                'source_dispatch_contract_if_source': 'actual paged BF16 D64 prefill metadata -> original optimized prefill64BF16 selector; original root/native image binding audited' if role == 'source' else None}
            rows.append(row)
            journal.emit('actual_fwd_enter', generation=generation, observation=row)
            returned = function(*args, **keywords)
            row['native_returned'] = True
            row['queue_after'] = queue_state(torch, arguments['q'].device)
            journal.emit('actual_fwd_return', generation=generation, observation=row)
            return returned

    return ObserveActualNativeCall()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'port'), required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=5)
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    precision.scope_guard()
    grant = None if args.cpu_only else check_grant(args)
    assert not args.output.exists()
    original, guards, canonical_preflight, local_path, verify_loaded, full_reference, tensor_digest = precision.shared_imports()
    manifest, manifest_path, package, required = canonical_preflight()
    assert manifest['source']['source_digest'] == precision.EXPECTED_DIGEST
    if args.role == 'port':
        hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
        required = set(hashes)
        native_binding = precision.verify_native_binding(package)
        compiler = {'whole_normal_package_icpx': True,
            'compiler_policy_sha256': digest(ART / precision.VERSION / 'compiler-policy-verified.json')}
    else:
        from matched_baseline_provenance import package_preflight, provenance
        package, hashes = package_preflight('source')
        compiler = provenance('source')
        imported = json.loads((ART / 'matched-native-icpx-v1/source/normal-import-verified.json').read_text())
        required = {Path(name).name for name in imported['loaded_normal_package_libraries_sha256']}
        native_binding = compiler['native_kernels']
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    assert not torch.xpu.is_initialized()
    verify_loaded(package, hashes, required)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    assert hasattr(torch._C._XpuStreamBase, 'sycl_queue')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    framework = original.import_framework(torch, args.cpu_only)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    plan = original.verify_plan()
    model = plan['cases'][0]
    assert model['name'] == CASE_NAME and model['serving_batch'] == 8
    golden_path = original.ROOT / 'source-accuracy.json'
    golden = next(row for row in json.loads(golden_path.read_text())['cases'] if row['name'] == CASE_NAME)
    bundle = torch.load(original.ROOT / model['bundle'], map_location='cpu', weights_only=True)
    case = SimpleNamespace(**model)
    report = {'role': args.role, 'runner_sha256': digest(Path(__file__)), 'case_name': CASE_NAME,
        'canonical_manifest_sha256': digest(manifest_path), 'source_digest': precision.EXPECTED_DIGEST,
        'SOURCE_golden_raw_sha256': digest(golden_path), 'normal_package': str(package),
        'native_binding': native_binding, 'mapped_native_SHA': {name: hashes[name] for name in sorted(required)},
        'compiler_scope_and_confound': compiler, 'no_timing_warmup_events': True,
        'capture8_per_generation': CAPTURE_COUNT, 'generations': GENERATIONS, 'full89_acceptance': False,
        'snapshot_instrumentation_boundary': 'full CPU snapshots outside capture alter diagnostic scheduling; inside capture only native scalar/address/queue API plus fsynced host markers, no GPU tensor readback',
        'capture_retention_policy': 'forensic-only strong references to all eight outputs until snapshots/full reference and explicit cleanup; declared lifecycle change, not production performance equivalence', 'cases': []}
    if args.cpu_only:
        tensors = original.fixture(model, bundle, torch, 'cpu')
        assert [tensor_digest(tensor, torch) for tensor in tensors] == golden['input_sha256']
        query, key, value, table, lengths, offsets = tensors
        inspect.signature(flash_attn_with_kvcache).bind(q=query, k_cache=key, v_cache=value,
            page_table=table, cache_seqlens=lengths, cu_seqlens_q=offsets, max_seqlen_q=4096,
            softmax_scale=0.125, causal=True, window_size=(2047, 0), softcap=0.0, return_softmax_lse=False)
        assert not torch.xpu.is_initialized()
        verify_loaded(package, hashes, required)
        report.update({'CPU_gate': True, 'GPU_initialized': False, 'input_sha256': golden['input_sha256'],
            'SOURCE_full_output_sha256': golden['output_sha256'], 'real_sycl_queue_API_descriptor_exists': True,
            'live_queue_address_observed': False, 'actual_exit': 0})
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'CPU_gate': 0, 'role': args.role, 'maps': len(required), 'GPU_initialized': False, 'queue_API_descriptor': True}))
        return
    assert digest(manifest_path) == grant['canonical_manifest_sha256']
    assert os.environ['ZE_AFFINITY_MASK'] == str(args.gpu_id) and os.sched_getaffinity(0) == set(range(16, 32))
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    folder = args.output.with_suffix('')
    folder.mkdir(exist_ok=False)
    journal = Journal(folder / 'protocol-markers.jsonl')
    journal.emit('normal_import_maps_verified', role=args.role, queue=queue_state(torch), native_binding=native_binding,
        mapped_native_SHA=report['mapped_native_SHA'], actual_registered_schema=str(torch.ops.sgl_kernel.fwd.default._schema))
    report['grant_sha256'] = digest(args.grant)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    entry = local_path(candidate['destination']) / 'src/sycl/xe35/fmha/entry.cpp'
    entry_text = entry.read_text()
    with torch.inference_mode():
        for generation in range(1, GENERATIONS + 1):
            torch.xpu.synchronize()
            time.sleep(2)
            journal.emit('generation_begin', generation=generation, queue=queue_state(torch))
            tensors = original.fixture(model, bundle, torch, 'xpu')
            assert [tensor_digest(tensor, torch) for tensor in tensors] == golden['input_sha256']
            holder = {}
            snapshot_stage('before_eager', generation, tensors, None, torch, folder, journal, native_binding, 'not produced')
            journal.emit('eager_begin', generation=generation, queue=queue_state(torch))
            holder['output'] = original.backend_call(model, tensors, torch, framework)
            torch.xpu.synchronize()
            journal.emit('eager_returned', generation=generation, queue=queue_state(torch))
            snapshot_stage('eager', generation, tensors, holder['output'], torch, folder, journal, native_binding, 'eager output')
            journal.emit('full_reference_begin', generation=generation, output_stage='eager')
            reference = full_reference(case, (*tensors, holder['output']), torch)
            actual = holder['output'].detach().cpu().contiguous()
            assert reference['reference_PASS'] and tensor_digest(actual, torch) == golden['output_sha256']
            journal.emit('full_reference_PASS', generation=generation, output_stage='eager', reference=reference,
                SOURCE_wholebits=True, output_sha256=tensor_digest(actual, torch))
            del actual
            snapshot_stage('before_capture', generation, tensors, holder['output'], torch, folder, journal, native_binding, 'verified eager output')
            rows, private_rows = [], []
            port_observer = precision.native_route_observer(torch, guards, entry_text, private_rows) if args.role == 'port' else None
            observer = capture_observer(torch, args.role, journal, generation, rows)
            capture_outputs = []
            graph = torch.xpu.XPUGraph()
            graph.enable_debug_mode()
            journal.emit('capture_begin', generation=generation, queue=queue_state(torch), capture_count=CAPTURE_COUNT)
            with observer:
                if port_observer is not None:
                    with port_observer:
                        with torch.xpu.graph(graph):
                            for call in range(CAPTURE_COUNT):
                                journal.emit('capture_launch_begin', generation=generation, call=call + 1, queue=queue_state(torch))
                                holder['output'] = original.backend_call(model, tensors, torch, framework)
                                capture_outputs.append(holder['output'])
                                journal.emit('capture_launch_returned', generation=generation, call=call + 1,
                                    output_tensor=tensor_metadata(holder['output']), queue=queue_state(torch))
                            journal.emit('before_capture_end', generation=generation, queue=queue_state(torch), actual_fwd_calls=rows)
                else:
                    with torch.xpu.graph(graph):
                        for call in range(CAPTURE_COUNT):
                            journal.emit('capture_launch_begin', generation=generation, call=call + 1, queue=queue_state(torch))
                            holder['output'] = original.backend_call(model, tensors, torch, framework)
                            capture_outputs.append(holder['output'])
                            journal.emit('capture_launch_returned', generation=generation, call=call + 1,
                                output_tensor=tensor_metadata(holder['output']), queue=queue_state(torch))
                        journal.emit('before_capture_end', generation=generation, queue=queue_state(torch), actual_fwd_calls=rows)
            journal.emit('capture_end_returned', generation=generation, queue=queue_state(torch), actual_fwd_calls=rows, private_route_observations=private_rows)
            assert len(rows) == CAPTURE_COUNT and all(row['native_returned'] for row in rows)
            if args.role == 'port':
                assert len(private_rows) == CAPTURE_COUNT and all(row['guard_evaluation_from_actual_metadata']['optimized'] for row in private_rows)
            torch.xpu.synchronize()
            path = folder / f'g{generation}.dot'
            journal.emit('debug_dump_begin', generation=generation, path=str(path))
            graph.debug_dump(str(path))
            assert path.is_file()
            journal.emit('debug_dump_returned', generation=generation, actual_DOT_sha256=digest(path))
            assert len(capture_outputs) == CAPTURE_COUNT
            snapshot_stage('before_replay', generation, tensors, capture_outputs, torch, folder, journal, native_binding,
                'capture-created output has not been executed; saved raw bytes only, no precision claim')
            journal.emit('replay_begin', generation=generation, queue=queue_state(torch))
            graph.replay()
            torch.xpu.synchronize()
            journal.emit('replay_returned', generation=generation, queue=queue_state(torch))
            snapshot_stage('after_replay', generation, tensors, capture_outputs, torch, folder, journal, native_binding, 'all eight executed replay outputs')
            replay_records = []
            for output_index, captured_output in enumerate(capture_outputs):
                journal.emit('full_reference_begin', generation=generation, output_stage='replay', output_index=output_index)
                reference = full_reference(case, (*tensors, captured_output), torch)
                actual = captured_output.detach().cpu().contiguous()
                assert reference['reference_PASS'] and tensor_digest(actual, torch) == golden['output_sha256']
                replay_record = {'output_index': output_index, 'full_reference': reference,
                    'SOURCE_wholebits': True, 'output_sha256': tensor_digest(actual, torch)}
                replay_records.append(replay_record)
                journal.emit('full_reference_PASS', generation=generation, output_stage='replay', **replay_record)
                del actual
            del captured_output
            report['cases'].append({'generation': generation, 'name': CASE_NAME, 'all_eight_replay_outputs': replay_records,
                'eager_and_all8_replay_SOURCE_wholebits': True, 'actual_fwd_calls': rows, 'private_routes': private_rows,
                'actual_DOT_sha256': digest(path)})
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            snapshot_stage('before_cleanup', generation, tensors, capture_outputs, torch, folder, journal, native_binding, 'all eight verified replay outputs')
            torch.xpu.synchronize()
            journal.emit('cleanup_begin', generation=generation, queue=queue_state(torch))
            del graph, tensors, holder, observer, port_observer, capture_outputs
            journal.emit('cleanup_returned', generation=generation, queue=queue_state(torch))
    verify_loaded(package, hashes, required)
    report['forensic_generations_PASS'] = len(report['cases']) == GENERATIONS
    journal.emit('role_terminal_PASS', role=args.role, full89_acceptance=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'forensic firstfail: {error!r}', file=sys.stderr, flush=True)
        raise SystemExit(2)

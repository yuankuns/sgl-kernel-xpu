import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from protocol import METHODS, PLAN, REPEATS, SAMPLES, SLICE, WARMUP, code_hashes, derived, digest, graph_proof, load_plan, method_order, phase_plan, read, require, save, summarize, validate_release


def scope():
    for name, value in {'BATCH_SIZE': '8', 'BATCHES': '8', 'BATCH': '8', 'SERVING_BATCH': '8', 'SERVING_BATCH_SIZE': '8',
        'SGLANG_BATCH_SIZE': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        require(os.environ.get(name, value) == value, f'{name} must be {value}; B8 only before GPU')
    for name in ('SAMPLES', 'REPEATS', 'WARMUP', 'GRAPH', 'QK_SAMPLES', 'QK_REPEATS', 'QK_WARMUP'):
        require(name not in os.environ, f'No override of reviewed fixed protocol: {name}')


def local(path):
    path = str(path)
    if path.startswith('/root/') and not Path(path).exists():
        path = path.replace('/root/', '/home/syk/', 1)
    if path.startswith('/home/syk/') and not Path(path).exists():
        path = path.replace('/home/syk/', '/root/', 1)
    return Path(path)


def live_builders():
    names = {'ninja', 'make', 'cmake', 'icpx', 'icx', 'clang', 'clang++', 'clang-23', 'clang-24', 'gcc', 'g++', 'cc', 'c++', 'cc1', 'cc1plus', 'ld', 'ld.lld', 'lld', 'ocloc', 'pip', 'pip3', 'uv', 'apt', 'apt-get', 'dpkg', 'conda', 'mamba'}
    live = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            comm = (directory / 'comm').read_text().strip()
            args = (directory / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and (comm in names or (comm.startswith('python') and ' -m pip ' in args)):
            live[directory.name] = {'comm': comm, 'args': args}
    return live


def release(path, initial):
    plan, fields, cases = load_plan()
    value = read(path)
    validate_release(value, plan, fields, cases, code_hashes(), time.time(), os.environ.get('ZE_AFFINITY_MASK'), sorted(os.sched_getaffinity(0)), initial)
    evidence = local(value['drain_evidence_path'])
    require(digest(evidence) == value['drain_evidence_sha256'], 'Main drain SHA mismatch')
    drain = read(evidence)
    require(drain.get('actual_audit_exit') == 0 and drain.get('matching_live_compiler_processes') == {}
        and drain.get('matching_live_package_manager_processes') == {}, 'Fresh all compiler/package-manager empty proof required')
    require(value['start_epoch'] > drain.get('epoch', float('inf')), 'Release must follow drain epoch')
    require(not live_builders(), 'Live compiler/package-manager activity; preserve hold, no GPU work')
    for name, checksum in plan['framework_and_source_file_SHA'].items():
        require(digest(local(name)) == checksum, f'Frozen framework/host source changed: {name}')
    return value, cases


def stack():
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and any(token in Path(parts[5]).name for token in
            ('libtorch', 'libc10', 'libsycl', 'libur_', 'libze_', 'libumf', 'libmpi')):
            path = Path(parts[5]).resolve()
            result[str(path)] = digest(path)
    return result


def core_check(expected):
    actual = stack()
    normalized = {str(local(path).resolve()): checksum for path, checksum in expected.items()}
    require(actual == normalized, 'Actual complete loaded core stack differs; do not claim matching stack')
    return actual


def bits_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__('torch').uint16).numpy().tobytes()).hexdigest()


def validate_output(original, actual, inputs, expected, method):
    import torch

    require(torch.equal(actual.view(torch.uint16), (original if method == METHODS[2] else expected).view(torch.uint16)), 'Full QKVO uint16 output mismatch')
    require(torch.equal(actual[:, 9216:].view(torch.uint16), original[:, 9216:].view(torch.uint16)), 'Tail V16+O_gate128 bitwise mismatch')
    result = {'full_QKVO_bitwise': True, 'tail_bitwise': True, 'actual_output_uint16_sha256': bits_sha(actual)}
    if method != METHODS[2]:
        sys.path.insert(0, str(SLICE.parent / 'normal-v4-accuracy-only-v1'))
        from driver import cpu_reference

        violations, elements, maximum = cpu_reference(original, actual, inputs['positions'], inputs['query_weight'].float(), inputs['key_weight'].float())
        require(violations == 0, f'Original full CPU reference strictfail{violations}; no timing acceptance')
        result.update({'violations': violations, 'QK_elements': elements, 'max_abs': maximum, 'atol': 0.005, 'rtol': 0.005})
    else:
        result['reference_kind'] = 'full original input bits, reset-only does not perform QK norm'
    return result


def captured(torch, method, qkv, initial, actual_call, directory):
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    captured_calls = 0
    origin_stream = torch.xpu.current_stream()
    stream = torch.xpu.Stream()
    stream.wait_stream(origin_stream)
    with torch.xpu.graph(graph, stream=stream):
        for logical_call in range(REPEATS):
            require(torch.xpu.is_current_stream_capturing(), 'Actual capture must be active; no replay shim')
            qkv.copy_(initial)
            if method == METHODS[1]:
                actual_call()
            captured_calls += 1
    require(captured_calls == REPEATS, 'All32 reset/adapter logical calls must be captured')
    origin_stream.wait_stream(stream)
    graph.instantiate()
    raw_graph = graph.raw_xpu_graph()
    raw_exec = graph.raw_xpu_graph_exec()
    require(raw_graph and raw_exec, 'Real backend graph pointers required')
    directory.mkdir(parents=True, exist_ok=False)
    graph.debug_dump(str(directory / 'capture.dot'))
    dumps = sorted(directory.glob('capture.dot*'))
    require(dumps and all(path.is_file() and path.stat().st_size for path in dumps), 'Actual debug DOT unavailable; preserve capture failure')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU], record_shapes=True) as profiler:
        graph.replay()
        torch.xpu.synchronize()
    trace = directory / 'actual-replay.trace.json'
    profiler.export_chrome_trace(str(trace))
    proof = graph_proof('\n'.join(path.read_text() for path in dumps), read(trace), method)
    proof.update({'raw_backend_graph': raw_graph, 'raw_backend_exec': raw_exec, 'actual_captured_logical_calls': captured_calls,
        'debug_DOT_sha256': {path.name: digest(path) for path in dumps}, 'actual_replay_trace_sha256': digest(trace),
        'profiler_is_unmeasured_graph_proof_only': True})
    save(directory / 'proof.json', proof)
    return graph, proof


def phase(grant_path, phase_id):
    grant, cases = release(grant_path, False)
    entry = next((item for item in phase_plan(cases) if item['phase'] == phase_id), None)
    require(entry is not None, 'Only fixed all19/three BAAB phases allowed')
    directory = SLICE / 'runs' / grant['release_id'] / 'phases' / phase_id
    directory.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'Normal complete package import must precede device initialization')
    sys.path.insert(0, str(SLICE.parent / 'normal-v4-model-config-accuracy-v2'))
    from run_accuracy_r2 import normal_audit

    before = normal_audit(entry['role'])
    require(not torch.xpu.is_initialized(), 'Normal package import unexpectedly initialized XPU')
    import sglang.kernels.ops.attention.fused_qknorm_rope as adapter

    require(hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU unavailable; short-circuit before arch query')
    require(tuple(torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device())) == (3, 5), 'Actual strict3.5 arch only; no fake vendor/ID')
    before_stack = core_check(grant['expected_core_loaded_stack_sha256'])
    save(directory / 'runtime-start.json', {'pid': os.getpid(), 'role': entry['role'], 'normal': before, 'core_loaded_stack': before_stack,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'physical_gpu': grant['gpu'], 'epoch': time.time()})
    for case_index, name in enumerate(entry['cases']):
        require(not live_builders(), 'New compiler/package-manager activity; stop actual phase')
        case = next(item for item in cases if item['name'] == name)
        path = local(case['fixture']['path'])
        require(digest(path) == case['fixture']['sha256'], 'Unchanged full shared inputs required')
        require(digest(local(case['expected_SOURCE_output'])) == case['expected_SOURCE_output_sha256'], 'Saved strict SOURCE output SHA mismatch')
        inputs = torch.load(path, map_location='cpu', weights_only=True)
        original = inputs['original']
        require(original.dtype == torch.bfloat16 and original.shape == (case['case']['tokens'], 18432), 'Original BF16 QKVO shape required')
        require(inputs['positions'].dtype == (torch.int64 if case['group'] == 'actual-config-semantic-synthetic' else torch.int32), 'Never pre-cast original positions')
        for weight in ('query_weight', 'key_weight'):
            require(torch.equal(inputs[weight].float(), inputs[weight].to(torch.bfloat16).float()), 'BF16 weights must preserve original complete values')
        initial = original.to('xpu:0')
        qkv = initial.clone()
        positions = inputs['positions'].to('xpu:0')
        query_weight = inputs['query_weight'].to(device='xpu:0', dtype=torch.bfloat16)
        key_weight = inputs['key_weight'].to(device='xpu:0', dtype=torch.bfloat16)
        expected = torch.load(local(case['expected_SOURCE_output']), map_location='cpu', weights_only=True)['actual']
        parameters = case['parameters']

        def actual_call():
            adapter.fused_qk_norm_rope(qkv, parameters['num_heads_q'], parameters['num_heads_k'], parameters['tail_heads'],
                parameters['head_dim'], parameters['norm_eps'], query_weight, key_weight, parameters['rope_theta'], parameters['is_neox'],
                positions, factor=parameters['factor'], low=parameters['low'], high=parameters['high'],
                attention_factor=parameters['attention_factor'], rotary_dim=parameters['rotary_dim'])

        torch.xpu.synchronize()
        time.sleep(2)
        require(torch.equal(qkv.cpu().view(torch.uint16), original.view(torch.uint16)), 'Actual complete GPU input bits differ')
        actual_positions = positions.cpu()
        require(actual_positions.dtype == inputs['positions'].dtype and torch.equal(actual_positions, inputs['positions']), 'Actual GPU positions must preserve original dtype and complete values')
        actual_weights = {'query_weight': query_weight.cpu(), 'key_weight': key_weight.cpu()}
        for weight, actual_weight in actual_weights.items():
            require(torch.equal(actual_weight.view(torch.uint16), inputs[weight].to(torch.bfloat16).view(torch.uint16)), 'Actual complete GPU BF16 weight bits mismatch')
        save(directory / f'{name}-actual-inputs.json', {'fixture_sha256': case['fixture']['sha256'],
            'complete_QKVO_uint16_sha256': bits_sha(original), 'positions_dtype': str(actual_positions.dtype),
            'complete_positions_bytes_sha256': bits_sha(actual_positions),
            'complete_GPU_BF16_weight_uint16_sha256': {weight: bits_sha(value) for weight, value in actual_weights.items()},
            'actual_GPU_input_full_bits_checked': True})
        for warmup in range(WARMUP):
            qkv.copy_(initial)
            actual_call()
        torch.xpu.synchronize()
        validate_output(original, qkv.cpu(), inputs, expected, METHODS[0])
        graphs = {}
        proofs = {}
        for method in METHODS[1:]:
            graphs[method], proofs[method] = captured(torch, method, qkv, initial, actual_call, directory / name / method)
            validate_output(original, qkv.cpu(), inputs, expected, method)
        starts = [torch.xpu.Event(enable_timing=True) for logical_call in range(REPEATS)]
        ends = [torch.xpu.Event(enable_timing=True) for logical_call in range(REPEATS)]
        for method in METHODS:
            for warmup in range(WARMUP):
                if method == METHODS[0]:
                    qkv.copy_(initial)
                    actual_call()
                else:
                    graphs[method].replay()
            torch.xpu.synchronize()
        raw = {method: [] for method in METHODS}
        repeats = []
        totals = {method: [] for method in METHODS[1:]}
        orders = []
        for sample in range(SAMPLES):
            order = method_order(entry['round'], entry['slot'], case_index, sample)
            orders.append(order)
            for method in order:
                if method == METHODS[0]:
                    for logical_call in range(REPEATS):
                        qkv.copy_(initial)
                        starts[logical_call].record()
                        actual_call()
                        ends[logical_call].record()
                    ends[-1].synchronize()
                    elapsed = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
                    repeats.append(elapsed)
                    raw[method].append(statistics.mean(elapsed))
                else:
                    starts[0].record()
                    graphs[method].replay()
                    ends[0].record()
                    ends[0].synchronize()
                    total = starts[0].elapsed_time(ends[0]) * 1000
                    totals[method].append(total)
                    raw[method].append(total / REPEATS)
                save(directory / name / 'raw-progress.json', {'phase': phase_id, 'case': name,
                    'fixture_sha256': case['fixture']['sha256'], 'raw_us': raw,
                    'per_call_raw_repeat_us': repeats, 'graph_raw_total_us': totals, 'sample_method_orders': orders})
                require(all(math.isfinite(item) and item > 0 for item in (elapsed if method == METHODS[0] else [total])),
                    'Invalid actual event value preserved in raw-progress; no filtered samples')
        correctness = {}
        for method in METHODS:
            if method == METHODS[0]:
                qkv.copy_(initial)
                actual_call()
            else:
                graphs[method].replay()
            torch.xpu.synchronize()
            actual = qkv.cpu()
            witness = None
            if entry['round'] == 0 and entry['slot'] in (0, 1):
                witness = SLICE / 'runs' / grant['release_id'] / 'witness' / entry['role'] / name / f'{method}.pt'
                witness.parent.mkdir(parents=True, exist_ok=True)
                torch.save({**inputs, 'actual': actual}, witness)
            try:
                correctness[method] = validate_output(original, actual, inputs, expected, method)
            except Exception:
                torch.save({**inputs, 'actual': actual}, directory / name / f'failure-{method}.pt')
                raise
            if witness is not None:
                correctness[method]['full_witness'] = {'path': str(witness), 'sha256': digest(witness)}
        result = {'phase': phase_id, 'case': name, 'group': case['group'], 'role': entry['role'], 'fixture_sha256': case['fixture']['sha256'],
            'position_original_dtype': str(inputs['positions'].dtype), 'methods': {method: summarize(raw[method]) for method in METHODS},
            'per_call_raw_repeat_us': repeats, 'graph_raw_total_us': totals, 'sample_method_orders': orders,
            'graph_capture_proofs': proofs, 'correctness': correctness, 'samples': SAMPLES, 'calls_per_sample': REPEATS,
            'difference_diagnostic_only': derived(raw[METHODS[1]], raw[METHODS[2]]),
            'actual_unchanged_framework_adapter_generic_dispatcher': True, 'pure_production_kernel_time_claim': False}
        save(directory / f'{name}.json', result)
        print(json.dumps({'phase': phase_id, 'case': name, 'role': entry['role'], 'all_three_columns_saved': True, 'all_output_ref_bits_checked': True}), flush=True)
        for graph in graphs.values():
            graph.reset()
        del graphs, proofs, actual, expected, inputs, original, qkv, initial, positions, query_weight, key_weight
    after = normal_audit(entry['role'])
    require(before['mapped_native_SHA'] == after['mapped_native_SHA'], 'Native maps changed during phase')
    after_stack = core_check(grant['expected_core_loaded_stack_sha256'])
    save(directory / 'runtime-end.json', {'pid': os.getpid(), 'normal': after, 'core_loaded_stack': after_stack, 'epoch': time.time()})


def suite(grant_path):
    grant, cases = release(grant_path, True)
    root = SLICE / 'runs' / grant['release_id']
    root.mkdir(parents=True, exist_ok=False)
    (root / 'grant-copy.json').write_bytes(grant_path.read_bytes())
    save(root / 'phase-plan.json', phase_plan(cases))
    save(root / 'supervisor.json', {'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'physical_gpu': grant['gpu'],
        'runner_files_sha256': code_hashes(), 'external_grant_sha256': digest(grant_path)})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    for entry in phase_plan(cases):
        prefix = root / entry['phase']
        arguments = [sys.executable, str(SLICE / 'runner.py'), '--mode', 'phase', '--grant', str(grant_path), '--phase', entry['phase']]
        save(prefix.with_suffix('.command.json'), arguments)
        with prefix.with_suffix('.log').open('w') as output:
            child = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT)
            save(prefix.with_suffix('.pid.json'), {'pid': child.pid})
            status = child.wait()
        prefix.with_suffix('.exit').write_text(f'{status}\n')
        require(status == 0, f'Actual phase exit{status}; preserve failure/no fallback/no retry')
    arguments = [sys.executable, str(SLICE / 'analyze.py'), '--run', str(root)]
    with (root / 'analysis.log').open('w') as output:
        status = subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT).returncode
    (root / 'analysis.exit').write_text(f'{status}\n')
    require(status == 0, f'Actual CPU analysis exit{status}')


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('cpu-check', 'suite', 'phase'))
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--phase')
    args = parser.parse_args()
    if args.mode == 'cpu-check':
        require(not args.grant and not args.phase, 'CPU preparation cannot consume any GPU grant')
        require(set(os.sched_getaffinity(0)) == set(range(16)), 'Current CPU-only assignment0–15')
        plan, fields, cases = load_plan()
        require('torch' not in sys.modules, 'Pure CPU contract check must not import/init Torch/XPU')
        print(json.dumps({'actual_exit': 0, 'cases': len(cases), 'phases': len(phase_plan(cases)), 'plan_SHA': fields['plan_SHA'],
            'actual_runner_files_SHA': code_hashes(), 'GPU_authorized': False, 'GPU_proof_NOT_executed': True}), flush=True)
    else:
        require(args.grant and args.grant.is_file(), 'New external main release REQUIRED before any Torch/GPU import')
        if args.mode == 'phase':
            require(args.phase, 'Fixed phase identifier required')
            phase(args.grant.resolve(), args.phase)
        else:
            require(not args.phase, 'Suite must run fixed all12 BAAB phases')
            suite(args.grant.resolve())


if __name__ == '__main__':
    main()

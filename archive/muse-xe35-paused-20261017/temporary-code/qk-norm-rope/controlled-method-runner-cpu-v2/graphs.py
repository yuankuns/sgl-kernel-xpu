import os
from pathlib import Path
import time

from protocol import METHODS, REPEATS, REPLAY_MARKER, digest, dot_structure, read, replay_proof, require, save


def profiler_state(torch):
    native = getattr(torch.autograd, '_profiler_enabled', None)
    require(callable(native), 'Real native profiler state API required')
    native_active = bool(native())
    python_active = bool(torch.autograd.profiler._is_profiler_enabled)
    require(not native_active and not python_active, 'Active profiler forbidden in timing process')
    instrumentation = {name: value for name, value in os.environ.items() if value and any(
        token in name for token in ('KINETO', 'PTI_', 'ZE_ENABLE_TRACING', 'ZE_ENABLE_API_TRACING', 'ZET_ENABLE_API_TRACING', 'UR_ENABLE_LAYERS', 'UR_TRACE', 'SYCL_PI_TRACE', 'SYCL_UR_TRACE'))}
    require(not instrumentation, 'Inherited tracing instrumentation requires separate review; no timing')
    return {'native_profiler_enabled': native_active, 'Python_profiler_enabled': python_active,
        'inherited_tracing_switches': instrumentation, 'timing_process_never_started_profiler': True}


def isolated_timing(torch, root, phase):
    require('QK_V2_PROOF_PROCESS' not in os.environ, 'Proof process cannot measure')
    prefix = root / (phase + '-proof')
    require(prefix.with_suffix('.exit').read_text().strip() == '0', 'True proof child terminal0 required')
    pid = read(prefix.with_suffix('.pid.json'))['pid']
    require(pid != os.getpid() and not Path(f'/proc/{pid}').exists(), 'Proof process must be absent before timing')
    return {**profiler_state(torch), 'proof_child_PID': pid, 'timing_PID': os.getpid(), 'proof_child_absent': True}


def capture_body(torch, method, graph, qkv, initial, actual_call):
    origin = torch.xpu.current_stream()
    stream = torch.xpu.Stream()
    stream.wait_stream(origin)
    with torch.xpu.graph(graph, stream=stream):
        for logical_call in range(REPEATS):
            require(torch.xpu.is_current_stream_capturing(), 'Real nativegraph capture required')
            qkv.copy_(initial)
            if method == METHODS[1]:
                actual_call()
    origin.wait_stream(stream)


def graph_metadata(graph, directory, method):
    graph.instantiate()
    pointers = {'raw_backend_graph': graph.raw_xpu_graph(), 'raw_backend_exec': graph.raw_xpu_graph_exec()}
    require(all(pointers.values()), 'Real graph/exec backend pointers required')
    graph.debug_dump(str(directory / 'capture.dot'))
    dumps = sorted(directory.glob('capture.dot*'))
    require(dumps and all(path.stat().st_size for path in dumps), 'Actual DOT required')
    return {**pointers, **dot_structure('\n'.join(path.read_text() for path in dumps), method),
        'debug_DOT_sha256': {path.name: digest(path) for path in dumps}, 'actual_captured_logical_calls': REPEATS,
        'pid': os.getpid(), 'starttick': Path('/proc/self/stat').read_text().split()[21]}


def output_proof(torch, qkv, initial, inputs, expected, method, directory):
    from runner import bits_sha, validate_output

    actual = qkv.cpu()
    torch.save({**inputs, 'actual': actual}, directory / 'full-output.pt')
    require(torch.equal(initial.cpu().view(torch.uint16), inputs['original'].view(torch.uint16)), 'Captured immutable original input changed')
    correctness = validate_output(inputs['original'], actual, inputs, expected, method)
    save(directory / 'whole-output.json', {**correctness, 'complete_witness_sha256': digest(directory / 'full-output.pt'),
        'initial_original_bits_unchanged': True, 'actual_output_uint16_sha256': bits_sha(actual)})
    return correctness


def sentinel(torch, qkv, directory):
    from runner import bits_sha

    qkv.fill_(float('nan'))
    torch.xpu.synchronize()
    actual = qkv.cpu()
    require(torch.isnan(actual).all().item(), 'Complete sentinel allNaN required')
    torch.save(actual, directory / 'sentinel.pt')
    save(directory / 'sentinel.json', {'allNaN': True, 'bits_sha256': bits_sha(actual), 'file_sha256': digest(directory / 'sentinel.pt')})


def raw_events(profiler):
    records = []
    require(profiler.profiler.kineto_results is not None, 'Actual Kineto results required')
    for event in profiler.profiler.kineto_results.events():
        record = {}
        for name in ('name', 'device_type', 'device_index', 'start_ns', 'end_ns', 'duration_ns', 'correlation_id', 'linked_correlation_id'):
            getter = getattr(event, name, None)
            if callable(getter):
                try:
                    value = getter()
                    record[name] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
                except Exception as error:
                    record[name + '_error'] = repr(error)
        records.append(record)
    return records


def proof_graph(torch, method, qkv, initial, actual_call, directory, inputs, expected, bindings):
    os.environ['QK_V2_PROOF_PROCESS'] = '1'
    directory.mkdir(parents=True, exist_ok=False)
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU], record_shapes=True)
    completed = False
    try:
        with profiler:
            with torch.profiler.record_function('QK_V2_PROOF_CAPTURE_ONLY'):
                capture_body(torch, method, graph, qkv, initial, actual_call)
            with torch.profiler.record_function('QK_V2_PROOF_INSTANTIATE_ONLY'):
                metadata = graph_metadata(graph, directory, method)
            with torch.profiler.record_function('QK_V2_PROOF_SENTINEL_ONLY'):
                sentinel(torch, qkv, directory)
            with torch.profiler.record_function(REPLAY_MARKER):
                graph.replay()
                torch.xpu.synchronize()
        completed = True
    finally:
        try:
            profiler.export_chrome_trace(str(directory / 'full-window.trace.json'))
            save(directory / 'raw-Kineto-events.json', raw_events(profiler))
        except Exception as error:
            save(directory / 'collection-error.json', {'error': repr(error), 'body_completed': completed})
            if completed:
                raise
    correctness = output_proof(torch, qkv, initial, inputs, expected, method, directory)
    dot = '\n'.join((directory / name).read_text() for name in metadata['debug_DOT_sha256'])
    proof = replay_proof(dot, read(directory / 'full-window.trace.json'), method)
    graph.reset()
    torch.xpu.synchronize()
    require(not torch.autograd._profiler_enabled() and not torch.autograd.profiler._is_profiler_enabled, 'Profiler must exit before proof graph reset completion')
    save(directory / 'proof.json', {**metadata, **proof, 'bindings': bindings, 'correctness': correctness,
        'actual_replay_trace_sha256': digest(directory / 'full-window.trace.json'),
        'profiler_exited_proofgraph_reset_synced': True, 'unmeasured_proof_only': True, 'epoch': time.time()})


def timing_graph(torch, method, qkv, initial, actual_call, directory, inputs, expected, bindings, proof_directory):
    state = profiler_state(torch)
    proof = read(proof_directory / 'proof.json')
    require(proof['profiler_exited_proofgraph_reset_synced'] and proof['pid'] != os.getpid(), 'Independent completed proof graph/process required')
    for name in ('phase', 'role', 'fixture_sha256', 'native_SHA', 'core_SHA'):
        require(proof['bindings'][name] == bindings[name], 'Proof/timing source/core/fixture binding differs')
    directory.mkdir(parents=True, exist_ok=False)
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    capture_body(torch, method, graph, qkv, initial, actual_call)
    metadata = graph_metadata(graph, directory, method)
    sentinel(torch, qkv, directory)
    graph.replay()
    torch.xpu.synchronize()
    correctness = output_proof(torch, qkv, initial, inputs, expected, method, directory)
    after = profiler_state(torch)
    result = {**metadata, 'bindings': bindings, 'correctness': correctness, 'profiler_before': state, 'profiler_after': after,
        'proofgraph_path': str(proof_directory), 'proofgraph_sha256': digest(proof_directory / 'proof.json'),
        'fresh_unprofiled_timinggraph': True, 'timinggraph_device_trace_NOT_collected': True,
        'DOT_and_whole_output_NOT_fake_device_trace': True, 'proof_and_timing_pointer_identity_is_PID_scoped': True}
    save(directory / 'proof.json', result)
    return graph, result

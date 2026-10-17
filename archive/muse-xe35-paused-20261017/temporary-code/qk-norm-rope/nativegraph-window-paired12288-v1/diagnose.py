import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from contract import MARKER, ROLES, ROOT, case, digest, hashes, local, read, require, save, scope, strict, unchanged, validate
from runner import bits_sha, core_check, live_builders, stack, validate_output


def normal(role):
    sys.path.insert(0, str(ROOT.parent / 'normal-v4-model-config-accuracy-v2'))
    from run_accuracy_r2 import normal_audit

    return normal_audit(role)


def cpu(role):
    require(sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Full CPU provider/input proof on assigned runtime CPUs32–47')
    unchanged()
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU provider audit cannot initialize XPU')
    audited = normal(role)
    require(not torch.xpu.is_initialized(), 'Normal CPU package audit initialized XPU')
    plan, fields, fixture = case()
    require(digest(local(fixture['fixture']['path'])) == fixture['fixture']['sha256'], 'Original complete inputs SHA')
    require(digest(local(fixture['expected_SOURCE_output'])) == fixture['expected_SOURCE_output_sha256'], 'Original golden full outputs SHA')
    expected = read(ROOT.parent / 'controlled-gpu6-release-v1/main-controlled-diagnostic-grant.json')['expected_core_loaded_stack_sha256']
    for path, checksum in expected.items():
        require(digest(local(path)) == checksum, 'Current actual core file SHA changed')
    core = stack()
    require(all(expected.get(path) == checksum for path, checksum in core.items()), 'Normal CPU mapped core differs from actual initialized binding')
    inputs = torch.load(local(fixture['fixture']['path']), map_location='cpu', weights_only=True, mmap=True)
    require(inputs['original'].dtype == torch.bfloat16 and inputs['original'].shape == (12288, 18432), 'Original BF16 full QKVO')
    require(inputs['positions'].dtype == torch.int64 and torch.equal(inputs['positions'], torch.arange(4096).repeat(3)), 'Original repeated positions semantics')
    profiler_libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and 'libpti' in Path(parts[5]).name:
            path = Path(parts[5]).resolve()
            profiler_libraries[str(path)] = digest(path)
    save(ROOT / f'CPU-provider-{role}.json', {'actual_exit': 0, 'pid': os.getpid(), 'XPU_initialized': False,
        'normal': audited, 'fixture': fixture, 'actual_CPU_core_maps_sha256': core,
        'current_initialized_core_files_sha256_verified': expected, 'actual_CPU_profiler_libraries_sha256': profiler_libraries,
        'original_full_inputs_and_golden_file_SHA_verified': True, 'frozen_code_sha256': hashes()})
    print(f'CPU full normal {role}/original fixture+golden/current core files actual0, XPU uninitialized', flush=True)


def native_events(profiler):
    results = profiler.profiler.kineto_results
    require(results is not None, 'Actual raw Kineto results required')
    records = []
    for event in results.events():
        record = {}
        for name in ('name', 'device_type', 'device_index', 'start_ns', 'end_ns', 'duration_ns', 'correlation_id',
            'linked_correlation_id', 'is_async', 'scope', 'device_resource_id', 'thread_id'):
            value = getattr(event, name, None)
            if callable(value):
                try:
                    output = value()
                    record[name] = output if isinstance(output, (str, int, float, bool, type(None))) else str(output)
                except Exception as error:
                    record[name + '_getter_error'] = repr(error)
        records.append(record)
    return records


def phase(grant_path, role):
    grant = read(grant_path)
    validate(grant, False)
    unchanged()
    require(os.environ.get('ZE_AFFINITY_MASK') == '6' and sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Physical runtime assignment')
    require(not live_builders(), 'No compiler/package manager during diagnostic')
    output = ROOT / 'run' / role
    output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'Normal import before XPU initialization')
    before = normal(role)
    require(not torch.xpu.is_initialized(), 'Normal package import initialized XPU')
    import sglang.kernels.ops.attention.fused_qknorm_rope as adapter

    require(torch.xpu.is_available(), 'XPU unavailable before arch query')
    require(tuple(torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device())) == (3, 5), 'Actual3.5 only')
    core = core_check(grant['core_loaded_stack_sha256'])
    plan, fields, fixture = case()
    require(digest(local(fixture['fixture']['path'])) == fixture['fixture']['sha256'], 'Full original input SHA changed')
    require(digest(local(fixture['expected_SOURCE_output'])) == fixture['expected_SOURCE_output_sha256'], 'Golden output SHA changed')
    inputs = torch.load(local(fixture['fixture']['path']), map_location='cpu', weights_only=True)
    original = inputs['original']
    expected = torch.load(local(fixture['expected_SOURCE_output']), map_location='cpu', weights_only=True)['actual']
    initial = original.to('xpu:0')
    qkv = initial.clone()
    positions = inputs['positions'].to('xpu:0')
    query_weight = inputs['query_weight'].to(device='xpu:0', dtype=torch.bfloat16)
    key_weight = inputs['key_weight'].to(device='xpu:0', dtype=torch.bfloat16)
    parameters = fixture['parameters']

    def actual_call():
        adapter.fused_qk_norm_rope(qkv, parameters['num_heads_q'], parameters['num_heads_k'], parameters['tail_heads'],
            parameters['head_dim'], parameters['norm_eps'], query_weight, key_weight, parameters['rope_theta'], parameters['is_neox'],
            positions, factor=parameters['factor'], low=parameters['low'], high=parameters['high'],
            attention_factor=parameters['attention_factor'], rotary_dim=parameters['rotary_dim'])

    torch.xpu.synchronize()
    time.sleep(2)
    require(torch.equal(initial.cpu().view(torch.uint16), original.view(torch.uint16)), 'Actual complete immutable input bits')
    require(positions.cpu().dtype == inputs['positions'].dtype and torch.equal(positions.cpu(), inputs['positions']), 'Actual positions unchanged')
    for name, value in (('query_weight', query_weight), ('key_weight', key_weight)):
        require(torch.equal(value.cpu().view(torch.uint16), inputs[name].to(torch.bfloat16).view(torch.uint16)), 'Complete actual BF16 weight bits')
    for setup in range(5):
        qkv.copy_(initial)
        actual_call()
    torch.xpu.synchronize()
    setup_correctness = validate_output(original, qkv.cpu(), inputs, expected, 'per_call')
    save(output / 'runtime-start.json', {'epoch': time.time(), 'pid': os.getpid(), 'normal': before, 'actual_core_maps_sha256': core,
        'fixture_sha256': fixture['fixture']['sha256'], 'golden_sha256': fixture['expected_SOURCE_output_sha256'],
        'setup_correctness': setup_correctness, 'actual_complete_input_bits_checked': True, 'original_unmeasured_setup_calls': 5})
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    origin = torch.xpu.current_stream()
    stream = torch.xpu.Stream()
    stream.wait_stream(origin)
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU], record_shapes=True)
    profile_completed = False
    try:
        with profiler:
            with torch.profiler.record_function('QK_DIAG_CAPTURE_ONLY'):
                with torch.xpu.graph(graph, stream=stream):
                    for logical_call in range(32):
                        require(torch.xpu.is_current_stream_capturing(), 'Actual native graph capture required')
                        qkv.copy_(initial)
                        actual_call()
            origin.wait_stream(stream)
            with torch.profiler.record_function('QK_DIAG_INSTANTIATE_ONLY'):
                graph.instantiate()
                raw_graph = graph.raw_xpu_graph()
                raw_exec = graph.raw_xpu_graph_exec()
                require(raw_graph and raw_exec, 'Actual backend graph pointers')
                graph.debug_dump(str(output / 'capture.dot'))
            with torch.profiler.record_function('QK_DIAG_SENTINEL_PREP_ONLY'):
                qkv.fill_(float('nan'))
                torch.xpu.synchronize()
                sentinel = qkv.cpu()
                require(torch.isnan(sentinel).all().item(), 'Actual complete sentinel preparation required')
                torch.save(sentinel, output / 'sentinel-before-replay.pt')
                save(output / 'sentinel-before-replay.json', {'all_elements_NaN': True, 'full_bits_sha256': bits_sha(sentinel),
                    'full_saved_tensor_sha256': digest(output / 'sentinel-before-replay.pt')})
            with torch.profiler.record_function(MARKER):
                graph.replay()
                torch.xpu.synchronize()
        profile_completed = True
    finally:
        try:
            profiler.export_chrome_trace(str(output / 'full-window.trace.json'))
            raw = native_events(profiler)
            save(output / 'raw-Kineto-events.json', raw)
        except Exception as error:
            save(output / 'actual-collection-export-error.json', {'error': repr(error), 'profile_body_completed': profile_completed})
            if profile_completed:
                raise
    trace = read(output / 'full-window.trace.json')
    save(output / 'PTI-device-timestamps.json', {'raw_standalone_PTI_callback_buffers_available': False,
        'provenance': 'actual raw Kineto XPU/PTI backend device events plus Chrome export; no fabricated direct PTI callback records',
        'raw_Kineto_device_records': [record for record in raw if 'XPU' in str(record.get('device_type', ''))],
        'actual_Chrome_device_records': [event for event in trace['traceEvents'] if event.get('ph') == 'X' and any(token in event.get('cat', '').lower() for token in ('kernel', 'gpu_memcpy', 'xpu_memcpy', 'gpu_memset'))]})
    actual = qkv.cpu()
    torch.save({**inputs, 'actual': actual}, output / 'complete-input-output-witness.pt')
    correctness = validate_output(original, actual, inputs, expected, 'captured_reset_adapter')
    require(torch.equal(initial.cpu().view(torch.uint16), original.view(torch.uint16)), 'Sentinel must not modify original captured initial input')
    save(output / 'sentinel-full-output.json', {**correctness, 'sentinel_preparation_complete': True,
        'full_witness_sha256': digest(output / 'complete-input-output-witness.pt'), 'unchanged_captured_initial_bits': True,
        'actual_output_uint16_sha256': bits_sha(actual), 'no_measured_samples_or_performance_events': True})
    dot_paths = sorted(output.glob('capture.dot*'))
    require(dot_paths, 'Actual DOT required')
    proof = strict('\n'.join(path.read_text() for path in dot_paths), trace)
    save(output / 'strict-replay-proof.json', {**proof, 'real_backend_graph': raw_graph, 'real_backend_exec': raw_exec,
        'captured_calls': 32, 'complete_output_reference_bits': correctness})
    after = normal(role)
    require(after['mapped_native_SHA'] == before['mapped_native_SHA'], 'Actual native maps changed')
    core_check(grant['core_loaded_stack_sha256'])
    save(output / 'runtime-end.json', {'epoch': time.time(), 'normal': after, 'strict_device_trace_and_full_accuracy_actual_exit': 0})
    graph.reset()
    print(f'{role} genuine32-reset-adapter graph/full-window real replay device trace/full reference5e-3+golden bits actual0', flush=True)


def suite(grant_path):
    grant = read(grant_path)
    validate(grant, True)
    require(os.environ.get('ZE_AFFINITY_MASK') == '6' and sorted(os.sched_getaffinity(0)) == list(range(32, 48)), 'Suite exact runtime assignment')
    output = ROOT / 'run'
    output.mkdir(exist_ok=False)
    save(output / 'supervisor.json', {'pid': os.getpid(), 'epoch': time.time(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'GPU': 6, 'driver_sha256': hashes(), 'external_grant_sha256': digest(grant_path)})
    time.sleep(max(0, grant['start_epoch'] - time.time()))
    for role in ROLES:
        command = [sys.executable, str(ROOT / 'diagnose.py'), '--mode', 'phase', '--role', role, '--grant', str(grant_path)]
        save(output / f'{role}.command.json', command)
        with (output / f'{role}.log').open('w') as logfile:
            child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
            save(output / f'{role}.pid.json', {'pid': child.pid})
            status = child.wait()
        (output / f'{role}.exit').write_text(f'{status}\n')
        require(status == 0, f'First actual {role} exit{status}; no next role/retry/proof relaxation')
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'Paired witness audit CPU-only')
    source = torch.load(output / 'source/complete-input-output-witness.pt', map_location='cpu', weights_only=True, mmap=True)
    port = torch.load(output / 'port/complete-input-output-witness.pt', map_location='cpu', weights_only=True, mmap=True)
    require(torch.equal(source['actual'].view(torch.uint16), port['actual'].view(torch.uint16)), 'Paired full SOURCE/PORT bits mismatch')
    save(output / 'PAIRED-COMPLETED.json', {'actual_exit': 0, 'strict_trace_and_reference_both_roles': True,
        'full_SOURCE_PORT_output_bitwise': True, 'no_performance_events_or_samples': True, 'final19_performance_scope_NOT_replaced': True})


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('cpu', 'phase', 'suite'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    if args.mode == 'cpu':
        require(args.role and not args.grant, 'CPU normal provider proof only')
        cpu(args.role)
    else:
        require(args.grant and args.grant.is_file(), 'New explicit grant required before Torch/GPU import')
        if args.mode == 'phase':
            require(args.role, 'Actual paired role required')
            phase(args.grant.resolve(), args.role)
        else:
            require(not args.role, 'Exactly both roles once')
            suite(args.grant.resolve())


if __name__ == '__main__':
    main()

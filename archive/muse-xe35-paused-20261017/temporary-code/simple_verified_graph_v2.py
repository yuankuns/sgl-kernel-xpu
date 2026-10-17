import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path


CALLS = 128
SAMPLES = 101
WARMUP = 5


def digest_file(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_parameters(*, calls=CALLS, samples=SAMPLES, warmup=WARMUP, serving_batch=8):
    for name, actual, expected in (('calls', calls, CALLS), ('samples', samples, SAMPLES), ('warmup', warmup, WARMUP), ('serving_batch', serving_batch, 8)):
        if type(actual) is not int or actual != expected:
            raise ValueError(f'{name} must be exactly {expected}')
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise ValueError(f'B8-only: {name} must be 8 before GPU work')
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            raise ValueError(f'{name} must be {expected} before GPU work')


def parse_dot_nodes(text):
    identifier = r'"(?:\\.|[^"\\])*"|[A-Za-z_\200-\377][A-Za-z_0-9\200-\377]*|(?:-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))'
    token_pattern = re.compile(r'/\*.*?\*/|//[^\n]*|\#[^\n]*|' + identifier + r'|->|--|[{}\[\];=,:]', re.DOTALL)
    tokens = [match.group() for match in token_pattern.finditer(text) if not match.group().startswith(('/*', '//', '#'))]
    if not tokens or 'digraph' not in tokens[:3] or '{' not in tokens or '}' not in tokens:
        raise ValueError('A complete native directed DOT graph is required')
    nodes = set()
    depth = 0
    for index, token in enumerate(tokens):
        if token == '[':
            if depth == 0 and index:
                name = tokens[index - 1]
                previous = tokens[index - 2] if index > 1 else None
                if re.fullmatch(identifier, name) and name not in ('graph', 'node', 'edge') and previous not in ('->', '--', '=', ':'):
                    identity = name[1:-1] if name.startswith('"') else name
                    if identity in nodes:
                        raise ValueError('Duplicate DOT node declaration: ' + identity)
                    nodes.add(identity)
            depth += 1
        elif token == ']':
            depth -= 1
            if depth < 0:
                raise ValueError('Unbalanced DOT attributes')
    if depth:
        raise ValueError('Unbalanced DOT attributes')
    return sorted(nodes)


def validate_node_ratio(single_nodes, graph_nodes, calls=CALLS):
    validate_parameters(calls=calls)
    if type(single_nodes) is not int or single_nodes <= 0:
        raise ValueError('Actual single-call DOT must contain at least one native node')
    if type(graph_nodes) is not int or graph_nodes != calls * single_nodes:
        raise ValueError('Actual repeated DOT nodes must equal 128 * single-call nodes')


def _save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2) + '\n')


def capture_paths(directory, calls):
    if type(calls) is not int or calls not in (1, CALLS):
        raise ValueError('DOT paths are only for single-call or128-call capture')
    dot_path = Path(directory) / ('single-call.dot' if calls == 1 else 'calls-128.dot')
    proof_path = dot_path.with_name(dot_path.stem + '-proof.json')
    return dot_path, proof_path


def _capture(torch, function, directory, calls, reset, before_capture, after_capture):
    if reset is not None:
        reset()
    torch.xpu.synchronize()
    if before_capture is not None:
        before_capture(calls)
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    captured_calls = 0
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        for logical_call in range(calls):
            if not torch.xpu.is_current_stream_capturing():
                raise RuntimeError('Python call selection requires real XPU capture')
            function()
            captured_calls += 1
    if after_capture is not None:
        after_capture(calls)
    dot_path, proof_path = capture_paths(directory, calls)
    graph.debug_dump(str(dot_path))
    dumps = sorted(directory.glob(dot_path.name))
    evidence = {'captured_calls': captured_calls, 'logical_calls': calls, 'keep_graph': True, 'timing_events_created': False, 'DOT_files': {path.name: digest_file(path) for path in dumps if path.is_file()}, 'DOT_nodes': None}
    if len(dumps) != 1 or not dumps[0].is_file() or not dumps[0].stat().st_size:
        _save(proof_path, evidence)
        raise RuntimeError('One genuine nonempty native debug DOT dump required')
    nodes = parse_dot_nodes(dumps[0].read_text())
    evidence.update(DOT_path=str(dumps[0]), DOT_sha256=digest_file(dumps[0]), DOT_nodes=len(nodes), node_ids=nodes)
    _save(proof_path, evidence)
    if captured_calls != calls:
        raise RuntimeError('Actual graph Python capture-call count mismatch')
    return graph, evidence


def verify_proof_files(proof):
    for label in ('single', 'repeated'):
        evidence = proof[label]
        path = Path(evidence['DOT_path'])
        if digest_file(path) != evidence['DOT_sha256']:
            raise RuntimeError('Native DOT evidence changed before timing')
        if len(parse_dot_nodes(path.read_text())) != evidence['DOT_nodes']:
            raise RuntimeError('Native DOT node proof changed before timing')
    validate_node_ratio(proof['single']['DOT_nodes'], proof['repeated']['DOT_nodes'])


def time_verified_graph(torch, function, *, case_index, case_name, evidence_directory, reset=None, check=None, check_single=None, before_capture=None, after_capture=None, calls=CALLS, samples=SAMPLES, warmup=WARMUP, serving_batch=8):
    validate_parameters(calls=calls, samples=samples, warmup=warmup, serving_batch=serving_batch)
    if type(case_index) is not int or case_index < 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case_name):
        raise ValueError('Nonnegative independent case index and safe case name required')
    if os.environ.get('ZE_AFFINITY_MASK') not in ('4', '5', '6', '7'):
        raise RuntimeError('New main performance grant on GPU4–7 required before GPU work')
    if check is None:
        raise ValueError('A real full-output post-replay accuracy check is required')
    directory = Path(evidence_directory) / f'case-{case_index:04d}-{case_name}'
    directory.mkdir(parents=True, exist_ok=False)
    torch.xpu.synchronize()
    time.sleep(2)
    for warmup_index in range(warmup):
        if reset is not None:
            reset()
        torch.xpu.synchronize()
        function()
    torch.xpu.synchronize()
    single_graph, single = _capture(torch, function, directory, 1, reset, before_capture, after_capture)
    if single['DOT_nodes'] <= 0:
        raise RuntimeError('Single-call native DOT has zero nodes; no repeated capture or events')
    single_graph.instantiate()
    if reset is not None:
        reset()
    torch.xpu.synchronize()
    single_graph.replay()
    torch.xpu.synchronize()
    (check_single if check_single is not None else check)()
    single_graph.reset()
    graph, repeated = _capture(torch, function, directory, calls, reset, before_capture, after_capture)
    proof = {'case_index': case_index, 'case': case_name, 'single': single, 'repeated': repeated, 'calls': calls, 'samples': samples, 'warmup': warmup, 'timing_events_created': False, 'reset_and_sync_outside_events': True}
    _save(directory / 'graph-proof.json', proof)
    verify_proof_files(proof)
    graph.instantiate()
    for warmup_index in range(warmup):
        if reset is not None:
            reset()
        torch.xpu.synchronize()
        graph.replay()
        torch.xpu.synchronize()
        check()
    raw_us = []
    raw_graph_ms = []
    try:
        with (directory / 'samples.jsonl').open('x') as output:
            for sample_index in range(samples):
                if reset is not None:
                    reset()
                torch.xpu.synchronize()
                verify_proof_files(proof)
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                raw_graph_ms.append(elapsed_ms)
                raw_us.append(elapsed_ms * 1000 / calls)
                output.write(json.dumps({'sample_index': sample_index, 'graph_ms': elapsed_ms, 'per_logical_call_us': raw_us[-1], 'calls': calls}) + '\n')
                output.flush()
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise RuntimeError('Every real device event duration must be finite and positive')
                check()
    finally:
        graph.reset()
    median = statistics.median(raw_us)
    result = {'case_index': case_index, 'case': case_name, 'calls': calls, 'samples': samples, 'warmup': warmup, 'p50_us': median, 'mad_us': statistics.median(abs(value - median) for value in raw_us), 'raw_us': raw_us, 'raw_graph_ms': raw_graph_ms, 'graph_proof': proof, 'graph_proof_path': str(directory / 'graph-proof.json'), 'graph_proof_sha256': digest_file(directory / 'graph-proof.json'), 'samples_path': str(directory / 'samples.jsonl'), 'samples_sha256': digest_file(directory / 'samples.jsonl'), 'accuracy': 'PASS', 'timing': '128 real captured logical operator calls; only graph replay inside device events; reset/copies/sync/checks excluded'}
    _save(directory / 'result.json', result)
    return result

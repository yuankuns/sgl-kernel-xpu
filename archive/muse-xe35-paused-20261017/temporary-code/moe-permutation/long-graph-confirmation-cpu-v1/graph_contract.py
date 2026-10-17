import hashlib
import json
import re
from pathlib import Path


CALLS = 1000
REPEATS = 41
WARMUP = 5


def count_dot_nodes(path):
    text = Path(path).read_text()
    nodes = set()
    pattern = r'^\s*("[^"\n]+"|[A-Za-z_][A-Za-z_0-9]*|[0-9]+)\s*\['
    for match in re.finditer(pattern, text, re.MULTILINE):
        name = match.group(1)
        if name not in ['node', 'edge', 'graph']:
            nodes.add(name)
    return len(nodes), hashlib.sha256(text.encode()).hexdigest()


def capture_verified_graph(torch, function, evidence_path, calls, warmup):
    if calls != CALLS or warmup != WARMUP:
        raise RuntimeError('Long-graph contract is exactly 1000 calls / warmup5')
    for _ in range(warmup):
        function()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph(keep_graph=True)
    graph.enable_debug_mode()
    actual_capture_calls = 0
    with torch.xpu.graph(graph):
        for _ in range(calls):
            function()
            actual_capture_calls += 1
    graph.debug_dump(str(evidence_path))
    actual_nodes, graph_hash = count_dot_nodes(evidence_path)
    proof = {'actual_capture_calls': actual_capture_calls, 'actual_native_DOT_nodes': actual_nodes,
             'DOT_sha256': graph_hash, 'calls': calls, 'warmup': warmup,
             'actual_normal_XPUGraph_capture': True, 'timing_started': False}
    Path(str(evidence_path) + '.json').write_text(json.dumps(proof, indent=2) + '\n')
    if actual_capture_calls != CALLS or actual_nodes != CALLS:
        raise RuntimeError('Actual XPUGraph must contain exactly 1000 nodes; stop before device timing')
    graph.instantiate()
    for _ in range(warmup):
        graph.replay()
    torch.xpu.synchronize()
    return graph

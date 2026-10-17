import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def native_calls(path):
    calls = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if ((isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == 'operators') or
            (isinstance(function, ast.Subscript) and isinstance(function.value, ast.Name) and function.value.id == 'operations')):
            calls.append(ast.dump(node, include_attributes=False))
    return sorted(calls)


def main():
    sys.path.insert(0, str(ROOT))
    import torch
    import simple_verified_graph_v1 as graph
    import simple_representative_performance_cases_v1 as cases

    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    assert not torch.xpu.is_initialized()
    cases.verify_frozen_factory()
    assert callable(graph.time_verified_graph) and callable(cases.representative_cases)
    for name in ('enable_debug_mode', 'debug_dump', 'instantiate', 'reset'):
        assert hasattr(torch.xpu.XPUGraph, name)
    drivers = (
        ('benchmark_memory_position_model_performance_v2.py', ['memory']),
        ('benchmark_logits_model_performance_v3.py', []),
        ('benchmark_simple_canonical_verified_perf_v2.py', ['memory']),
    )
    math_checks = {}
    for filename, positional in drivers:
        importlib.import_module(Path(filename).stem)
        before = ROOT / 'simple-model-performance-v2-cpu-preparation' / ('before-helper-integration-' + filename)
        current = ROOT / filename
        assert native_calls(current) == native_calls(before)
        math_checks[filename] = {'native_calls_unchanged': True, 'before_sha256': digest(before), 'current_sha256': digest(current)}
    actual_dot = ROOT / 'moe-permutation/formal-v4-perf-gpu4-longgraph-v6/canonical-v4-icpx-formal-gpu4-longgraph-v6-round1-BAAB-0-source-int32-active23-graph.dot'
    assert len(graph.parse_dot_nodes(actual_dot.read_text())) == 1000
    aliases = ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
               'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS')
    negatives = []
    for filename, positional in drivers:
        command = [sys.executable, '-B', str(ROOT / filename), *positional, '--label', 'before',
                   '--phase', 'performance', '--grant', '/nonexistent-main-grant', '--output', '/nonexistent-output.json']
        for alias in aliases:
            environment = {name: value for name, value in os.environ.items() if name not in aliases}
            environment.update({alias: '1', 'ZE_AFFINITY_MASK': '7'})
            result = subprocess.run(command, env=environment, text=True, capture_output=True)
            assert result.returncode == 2 and 'B8-only' in result.stderr and 'Traceback' not in result.stderr
            negatives.append({'driver': filename, 'alias': alias, 'actual_exit': result.returncode})
    launcher = ROOT / 'canonical-tools/run_simple_verified_performance_v2.py'
    result = subprocess.run([sys.executable, '-B', str(launcher)], text=True, capture_output=True)
    assert result.returncode == 2 and 'authorization required' in result.stderr
    assert not torch.xpu.is_initialized()
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_initialized': False, 'native_calls': math_checks,
              'helper_sha256': digest(ROOT / 'simple_verified_graph_v1.py'),
              'case_helper_sha256': digest(ROOT / 'simple_representative_performance_cases_v1.py'),
              'launcher_sha256': digest(launcher), 'before_Torch_negative_guards': negatives,
              'actual_saved_1000_node_DOT_parser_count': 1000, 'actual_DOT_sha256': digest(actual_dot),
              'native_graph_instantiation_support_or_timing_proven': False,
              'normal_six_provider_CPU_gates': 'separate full-normal-CPU-gates.exit0',
              'explicit_default_launcher_refusal_exit': result.returncode}
    with (ROOT / 'SIMPLE_INTEGRATED_PERFORMANCE_CPU_PREFLIGHT_v3.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'native_call_AST_pairs': len(math_checks),
                      'negative_guards': len(negatives), 'actual_DOT_nodes': 1000, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

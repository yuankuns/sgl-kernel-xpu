import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main_ast(text):
    tree = ast.parse(text)
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'), include_attributes=False)


def main():
    drivers = {
        'memory_position': ('benchmark_memory_position_model_performance_v2.py', ['memory']),
        'logits_semantic': ('benchmark_logits_model_performance_v3.py', []),
        'representative': ('benchmark_simple_canonical_verified_perf_v2.py', ['memory']),
    }
    unchanged = {}
    for old_name, new_name in (
        ('benchmark_memory_position_model_config_v1.py', 'benchmark_memory_position_model_performance_v2.py'),
        ('benchmark_logits_model_config_v2.py', 'benchmark_logits_model_performance_v3.py'),
    ):
        normalized = (ROOT / new_name).read_text().replace('from simple_verified_graph_v1 import time_graph', 'from benchmark_simple import time_graph')
        assert main_ast(normalized) == main_ast((ROOT / old_name).read_text())
        unchanged[new_name] = {'native_math_fixture_reference_main_AST_unchanged': True,
                               'original_sha256': digest(ROOT / old_name), 'current_sha256': digest(ROOT / new_name)}
    aliases = ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
               'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS')
    negatives = []
    for name, (filename, positional) in drivers.items():
        ast.parse((ROOT / filename).read_text())
        command = [sys.executable, '-B', str(ROOT / filename), *positional, '--label', 'before',
                   '--phase', 'performance', '--grant', '/nonexistent-main-grant', '--output', '/nonexistent-output.json']
        for alias in aliases:
            environment = {key: value for key, value in os.environ.items() if key not in aliases}
            environment.update({'ZE_AFFINITY_MASK': '7', alias: '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            result = subprocess.run(command, env=environment, text=True, capture_output=True)
            assert result.returncode == 2 and 'B8-only' in result.stderr
            assert 'ModuleNotFoundError' not in result.stderr and 'Traceback' not in result.stderr
            negatives.append({'driver': name, 'alias': alias, 'requested_batch': 1, 'actual_exit': result.returncode})
        environment = {key: value for key, value in os.environ.items() if key not in aliases}
        environment.update({'ZE_AFFINITY_MASK': '9', 'PYTHONDONTWRITEBYTECODE': '1'})
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        assert result.returncode == 2 and 'Traceback' not in result.stderr
        negatives.append({'driver': name, 'accuracy_only_card_rejected_for_performance': 9, 'actual_exit': result.returncode})
    assert 'torch' not in sys.modules and 'sgl_kernel' not in sys.modules
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_started': False,
              'torch_imported': False, 'native_math_checks': unchanged, 'negative_guards': negatives,
              'graph_helper_and_case_helper_runtime_validation_pending': True,
              'performance_runner_or_future_grant_created': False}
    with (ROOT / 'SIMPLE_VERIFIED_PERFORMANCE_CPU_PREFLIGHT_v2.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'negative_guards': len(negatives), 'native_main_AST_pairs': len(unchanged), 'GPU_started': False}))


if __name__ == '__main__':
    main()

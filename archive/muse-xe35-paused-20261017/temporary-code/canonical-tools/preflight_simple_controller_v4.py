import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    before = ROOT / 'canonical-tools/run_simple_verified_performance_v3.py'
    current = ROOT / 'canonical-tools/run_simple_verified_performance_v4.py'
    trees = [ast.parse(path.read_text()) for path in (before, current)]
    phase_functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'container_phase') for tree in trees]
    assert ast.dump(phase_functions[0], include_attributes=False) == ast.dump(phase_functions[1], include_attributes=False)
    result = subprocess.run([sys.executable, '-B', str(current)], capture_output=True, text=True)
    assert result.returncode == 2 and 'authorization required' in result.stderr and 'Traceback' not in result.stderr
    for name in ('ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1', 'SIMPLE_INTEGRATED_PERFORMANCE_CPU_PREFLIGHT_v4', 'MAINTENANCE_QUIET_AFTER_ROUTING_v4'):
        assert int((ROOT / (name + '.exit')).read_text()) == 0
    assert int((ROOT / 'simple-model-performance-v4-cpu-preparation/full-normal-CPU-gates.exit').read_text()) == 0
    report = {'actual_exit': 0, 'actual_default_launcher_refusal_exit': result.returncode,
              'container_phase_AST_unchanged': True,
              'controller_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
              'GPU_started': False, 'timing_acceptance': False}
    with (ROOT / 'SIMPLE_CONTROLLER_CPU_PREFLIGHT_v4.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

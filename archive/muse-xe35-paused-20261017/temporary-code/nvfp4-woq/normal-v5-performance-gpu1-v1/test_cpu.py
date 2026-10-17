import argparse
import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

import policy


class CPUContracts(unittest.TestCase):
    def test_syntax(self):
        for path in policy.DIRECTORY.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_no_math_helpers_changed(self):
        old = ast.parse((policy.ACCURACY / 'run_accuracy_case.py').read_text())
        new = ast.parse((policy.DIRECTORY / 'run_case.py').read_text())
        for name in ('tensor_hash_xe35', 'full_tensor_evidence'):
            first = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
            second = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(first), ast.dump(second))

    def test_original_representative_factory(self):
        original = ast.parse((policy.SOURCE / 'benchmark/bench_nvfp4_woq_group_gemm.py').read_text())
        body = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        assignments = [node for node in ast.walk(body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'shapes' for target in node.targets)]
        defaults = next(ast.literal_eval(node.value) for node in assignments if isinstance(node.value, ast.List) and len(node.value.elts) == 4)
        self.assertEqual(policy.REPRESENTATIVE, [(experts, experts * per_expert, channels, inner) for experts, per_expert, channels, inner in defaults])
        source = (policy.DIRECTORY / 'run_case.py').read_text()
        self.assertIn("original_factory.build_inputs(experts, rows // experts, channels, inner, 'xpu')", source)
        self.assertNotIn('def quantize_', source)

    def test_production_cases(self):
        self.assertEqual(policy.PRODUCTION, [(256, 64, 8192, 8192), (256, 262144, 8192, 8192), (256, 64, 8192, 4096), (256, 262144, 8192, 4096)])
        self.assertEqual(policy.ORDER, ('source', 'after', 'after', 'source'))

    def test_B8_negative_before_Torch(self):
        for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'ISL', 'OSL', 'TP', 'EP'):
            environment = dict(os.environ, **{name: '7'})
            process = subprocess.run([sys.executable, '-B', str(policy.DIRECTORY / 'run_case.py')], env=environment, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('before Torch/GPU', process.stderr)

    def test_no_grant_and_no_default_launch(self):
        environment = dict(os.environ)
        environment.pop('NV_V5_PERFORMANCE_GPU1_GRANT', None)
        for script, error in (('run_case.py', 'grant required before Torch/GPU'), ('suite.py', 'No default launch')):
            process = subprocess.run([sys.executable, '-B', str(policy.DIRECTORY / script)], env=environment, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(error, process.stderr)

    def test_guard_before_actual_Torch_import(self):
        tree = ast.parse((policy.DIRECTORY / 'run_case.py').read_text())
        torch_index = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names))
        guards = [index for index, node in enumerate(tree.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ('preflight', 'runtime_grant')]
        self.assertEqual(len(guards), 2)
        self.assertLess(max(guards), torch_index)

    def test_actual_Torch_CPU_only(self):
        import torch
        self.assertFalse(torch.xpu.is_initialized())
        self.assertEqual(torch.arange(8).device.type, 'cpu')
        self.assertEqual(set(os.sched_getaffinity(0)), set(range(80, 96)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    results = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUContracts))
    policy.save(args.output, {'actual_exit': 0 if results.wasSuccessful() else 1, 'tests': results.testsRun, 'failures': len(results.failures), 'errors': len(results.errors), 'XPU_initialized': False, 'GPU_started': False, 'compiler_started': False, 'CPU_affinity': sorted(os.sched_getaffinity(0))})
    raise SystemExit(0 if results.wasSuccessful() else 1)

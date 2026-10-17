import ast
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from forensic_runner import DIRECTORY, ROOT, digest, native_rows, pointers, save


class CPUProposalContracts(unittest.TestCase):
    def test_syntax_and_no_timer(self):
        for path in DIRECTORY.glob('*.py'):
            tree = ast.parse(path.read_text(), filename=str(path))
            if path.name != 'forensic_runner.py':
                continue
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
            self.assertNotIn('Event', attributes)
            self.assertNotIn('elapsed_time', attributes)
            self.assertIn('instantiate', attributes)
            self.assertIn('debug_dump', attributes)
            self.assertEqual(attributes.count('replay'), 1)

    def test_save_before_fullcheck(self):
        tree = ast.parse((DIRECTORY / 'forensic_runner.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'save_actual')
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        saved_line = next(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'save')
        check_line = min(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'assert_close')
        self.assertLess(saved_line, check_line)
        self.assertIn('atol=0, rtol=0', ast.get_source_segment((DIRECTORY / 'forensic_runner.py').read_text(), function))

    def test_actual_old_DOT_not_tensor_pointer_values(self):
        path = ROOT / 'simple-model-verified-performance-v4-gpu7-run1/round-0-slot-0-before/logits-vocab-semantic.graphs/case-0001-mask-decode8/single-call.dot'
        rows = native_rows(path.read_text())
        self.assertEqual(rows['arg[0]'], 'long* _arg_input')
        self.assertEqual(rows['arg[1]'], 'long* _arg_mapped')
        self.assertEqual(rows['arg[2]'], 'char* _arg_invalid')
        self.assertTrue(rows['pArguments'].startswith('0x'))
        self.assertNotEqual(rows['arg[0]'], rows['pArguments'])
        self.assertIn('VocabMaskKernel', rows['kernelName'])

    def test_no_default_launch(self):
        process = subprocess.run([sys.executable, '-B', str(DIRECTORY / 'forensic_runner.py')], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn('CPU proposal only', process.stderr)

    def test_bad_serving_scope_preimport(self):
        for name in ('BATCHES', 'ISL', 'OSL', 'TP', 'EP'):
            process = subprocess.run([sys.executable, '-B', str(DIRECTORY / 'forensic_runner.py')], env=dict(os.environ, **{name: '7'}), capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('before Torch/GPU', process.stderr)

    def test_actual_Torch_CPU_API_and_reference(self):
        import torch
        self.assertFalse(torch.xpu.is_initialized())
        for name in ('enable_debug_mode', 'debug_dump', 'instantiate', 'replay', 'reset'):
            self.assertTrue(callable(getattr(torch.xpu.XPUGraph, name)))
        ids = torch.tensor([-1, 0, 1, 202047, 202048, 100, 70000, 150000], dtype=torch.int64)
        invalid = (ids < 0) | (ids >= 202048)
        self.assertEqual((ids * (~invalid)).tolist(), [0, 0, 1, 202047, 0, 100, 70000, 150000])
        actual = pointers(ids)
        self.assertEqual(actual['data_ptr'], ids.data_ptr())
        self.assertEqual(actual['device'], 'cpu')
        self.assertFalse(torch.xpu.is_initialized())


if __name__ == '__main__':
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUProposalContracts))
    save(DIRECTORY / 'CPU-PROPOSAL-CONTRACT.json', {'actual_exit': 0 if result.wasSuccessful() else 1, 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'GPU_started': False, 'compiler_started': False, 'XPU_initialized': False, 'forensic_runner_SHA256': digest(DIRECTORY / 'forensic_runner.py'), 'test_SHA256': digest(__file__)})
    raise SystemExit(0 if result.wasSuccessful() else 1)

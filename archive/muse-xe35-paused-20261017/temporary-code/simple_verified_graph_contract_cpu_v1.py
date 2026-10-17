import argparse
import ast
import inspect
import json
import os
import sys
import unittest
from pathlib import Path

import torch

import simple_representative_performance_cases_v1 as cases
import simple_verified_graph_v1 as graph


ROOT = Path(__file__).resolve().parent


class CPUContracts(unittest.TestCase):
    def test_real_torch_graph_API_without_device_initialization(self):
        self.assertFalse(torch.xpu.is_initialized())
        self.assertIn('keep_graph', inspect.signature(torch.xpu.XPUGraph.__new__).parameters)
        for name in ('capture_begin', 'capture_end', 'enable_debug_mode', 'debug_dump', 'instantiate', 'replay', 'reset'):
            self.assertTrue(callable(getattr(torch.xpu.XPUGraph, name)))
        self.assertTrue(callable(torch.xpu.is_current_stream_capturing))

    def test_fixed_parameters_and_negative_types(self):
        graph.validate_parameters()
        for name, values in {'calls': (0, 1, 127, 129, True, 128.0), 'samples': (0, 100, 102, True, 101.0), 'warmup': (0, 4, 6, True, 5.0), 'serving_batch': (0, 7, 9, True, 8.0)}.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    graph.validate_parameters(**{name: value})

    def test_environment_rejected_without_GPU_work(self):
        for name, value in {'BATCHES': '7', 'BATCH': '7', 'SERVING_BATCH': '7', 'BATCH_SIZE': '7', 'SGLANG_BATCH_SIZE': '7', 'SGLANG_MUSE_BATCH_SIZE': '7', 'MUSE_BATCH_SIZE': '7', 'SERVING_BATCH_SIZE': '7', 'MUSE_BS': '7', 'BS': '7', 'ISL': '4095', 'OSL': '15', 'TP': '2', 'EP': '2'}.items():
            previous = os.environ.get(name)
            try:
                os.environ[name] = value
                with self.subTest(name=name), self.assertRaises(ValueError):
                    graph.validate_parameters()
            finally:
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        self.assertFalse(torch.xpu.is_initialized())

    def test_pure_stdlib_DOT_parser_fixtures_not_GPU_proof(self):
        text = '''strict digraph graph_name {
            graph [label="graph header"]; node [shape=box]; edge [color=blue];
            // fake [label="not a node"];
            /* other [label="not a node"]; */
            "node 1" [label="kernel [attribute] with \\"quoted\\" name"];
            node_2 [
                label="kernel second", shape=box
            ];
            3 [label="kernel third"];
            "node 1" -> node_2 [label="edge attributes"];
            node_2 -> 3;
        }'''
        self.assertEqual(graph.parse_dot_nodes(text), ['3', 'node 1', 'node_2'])
        self.assertEqual(graph.parse_dot_nodes('digraph empty {}'), [])
        self.assertEqual(graph.parse_dot_nodes('digraph G { node [label="default"]; }'), [])
        for invalid in ('', 'not a DOT graph', 'digraph G { same [label="first"]; same [label="duplicate"]; }', 'digraph G { bad [label="not closed"; }'):
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                graph.parse_dot_nodes(invalid)

    def test_native_node_ratio_negative_contracts(self):
        graph.validate_node_ratio(1, 128)
        graph.validate_node_ratio(3, 384)
        for single, repeated in ((0, 0), (-1, 128), (1, 127), (2, 128), (3, 385), (True, 128), (1, 128.0)):
            with self.subTest(single=single, repeated=repeated), self.assertRaises(ValueError):
                graph.validate_node_ratio(single, repeated)

    def test_no_performance_factory_on_accuracy_GPU(self):
        previous = os.environ.get('ZE_AFFINITY_MASK')
        try:
            for gpu in ('0', '3', '8', '9', '10', '11', '4,5'):
                os.environ['ZE_AFFINITY_MASK'] = gpu
                with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                    next(cases.representative_cases(torch, {}, 'memory'))
                with self.assertRaises(RuntimeError):
                    graph.time_verified_graph(torch, None, case_index=0, case_name='contract-negative', evidence_directory=ROOT, check=lambda: None)
        finally:
            if previous is None:
                os.environ.pop('ZE_AFFINITY_MASK', None)
            else:
                os.environ['ZE_AFFINITY_MASK'] = previous
        self.assertFalse(torch.xpu.is_initialized())

    def test_frozen_factories_and_reference_math_preserved(self):
        cases.verify_frozen_factory()
        self.assertEqual(graph.digest_file(ROOT / 'benchmark_simple.py'), cases.FROZEN_FACTORY_SHA256)
        self.assertEqual(graph.digest_file(ROOT / 'benchmark_simple_canonical_v1.py'), 'f900f7e6d593728af0f580491cf9309e317987e18830bf1b3563308891891723')
        source = inspect.getsource(cases.representative_cases)
        for name in ('benchmark_simple.memory_cases', 'benchmark_simple.position_cases', 'benchmark_simple.logits_cases'):
            self.assertIn(name, source)
        self.assertIn('not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch()', source)
        state = ast.parse(inspect.getsource(cases.IndependentSoftcapBuffers))
        initializer = next(node for node in ast.walk(state) if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        self.assertIn("reference['expected_logits']", inspect.getsource(cases.IndependentSoftcapBuffers.__init__))
        self.assertIn("inputs['original']", inspect.getsource(cases.IndependentSoftcapBuffers.__init__))
        self.assertIn('(8, 202048)', ast.unparse(initializer))
        for method in (cases.IndependentSoftcapBuffers.check_single, cases.IndependentSoftcapBuffers.check):
            checks = [node for node in ast.walk(ast.parse(inspect.getsource(method).strip())) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'assert_close']
            self.assertEqual(len(checks), 1)
            self.assertEqual({keyword.arg: ast.literal_eval(keyword.value) for keyword in checks[0].keywords}, {'atol': 3e-5, 'rtol': 3e-5})

    def test_AST_capture_has_only_real_calls_and_selector(self):
        capture = ast.parse(inspect.getsource(graph._capture))
        block = next(node for node in ast.walk(capture) if isinstance(node, ast.With))
        names = [ast.unparse(node.func) for node in ast.walk(block) if isinstance(node, ast.Call)]
        self.assertIn('function', names)
        self.assertIn('torch.xpu.is_current_stream_capturing', names)
        self.assertNotIn('reset', names)
        self.assertFalse(any(name.endswith('.copy_') for name in names))
        selector = ast.parse(inspect.getsource(cases.IndependentSoftcapBuffers.launch).strip())
        branch = next(node for node in ast.walk(selector) if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(branch.test), 'self.torch.xpu.is_current_stream_capturing()')
        self.assertTrue(any(isinstance(node, ast.AugAssign) and ast.unparse(node.target) == 'self.capture_index' for child in branch.body for node in ast.walk(child)))
        self.assertFalse(any(isinstance(node, ast.AugAssign) for child in branch.orelse for node in ast.walk(child)))
        resets = ast.parse(inspect.getsource(cases.IndependentSoftcapBuffers.reset).strip())
        loop = next(node for node in ast.walk(resets) if isinstance(node, ast.For))
        self.assertEqual(ast.unparse(loop.iter), 'self.buffers')
        self.assertIn('buffer.copy_(self.original)', ast.unparse(loop))

    def test_AST_proof_before_events_and_copies_outside_interval(self):
        source = ast.parse(inspect.getsource(graph.time_verified_graph))
        events = [node for node in ast.walk(source) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.xpu.Event']
        proofs = [node for node in ast.walk(source) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'verify_proof_files']
        self.assertEqual(len(events), 2)
        self.assertTrue(all(node.lineno > min(proof.lineno for proof in proofs) for node in events))
        sample_loop = next(node for node in ast.walk(source) if isinstance(node, ast.For) and ast.unparse(node.target) == 'sample_index')
        calls = [node for node in ast.walk(sample_loop) if isinstance(node, ast.Call)]
        start_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'start.record')
        end_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'end.record')
        between = [ast.unparse(node.func) for node in calls if start_line < node.lineno < end_line]
        self.assertEqual(between, ['graph.replay'])
        reset_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'reset')
        sync_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'torch.xpu.synchronize')
        self.assertLess(reset_line, min(node.lineno for node in events))
        self.assertLess(sync_line, min(node.lineno for node in events))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT / 'simple-verified-graph-contract-cpu-v1.json')
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != list(range(80, 96)):
        raise RuntimeError('CPU-only contract tests require CPU80–95')
    if torch.xpu.is_initialized():
        raise RuntimeError('XPU must remain uninitialized')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUContracts))
    if torch.xpu.is_initialized():
        raise RuntimeError('CPU contract suite initialized XPU unexpectedly')
    report = {'actual_CPU_contract_exit': 0 if result.wasSuccessful() else 1, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'tests_run': result.testsRun, 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__, 'Torch_imported': True, 'XPU_initialized': False, 'actual_XPUGraph_instantiated_or_captured': False, 'GPU_node_proof_passed': False, 'GPU_timing_started': False, 'synthetic_DOT_parser_fixtures_are_not_runtime_GPU_evidence': True, 'files_sha256': {name: graph.digest_file(ROOT / name) for name in ('simple_verified_graph_v1.py', 'simple_representative_performance_cases_v1.py', 'simple_verified_graph_contract_cpu_v1.py', 'benchmark_simple.py', 'benchmark_simple_canonical_v1.py')}}
    target = args.report
    with target.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report['actual_CPU_contract_exit']


if __name__ == '__main__':
    raise SystemExit(main())

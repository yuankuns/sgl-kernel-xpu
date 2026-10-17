import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from protocol import METHODS, ORDER, PAIRED, REPEATS, SAMPLES, SLICE, VERSION, code_hashes, derived, dot_structure, graph_proof, load_plan, method_order, paired_binding, phase_plan, read, replay_proof, REPLAY_MARKER, validate_release
from graphs import profiler_state


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.fields, cls.cases = load_plan()
        cls.runner = ast.parse((SLICE / 'runner.py').read_text())
        cls.graphs = ast.parse((SLICE / 'graphs.py').read_text())
        cls.old = SLICE.parent / 'controlled-method-runner-cpu-v1'

    def schema(self):
        return {'version': VERSION, 'main_authorized': True, 'release_id': 'unit_NOT_real_grant', 'gpu': 6,
            'stable_gpu_exclusive_owner': 'QK controlled diagnostic', 'runtime_cpu_owner': 'QK controlled diagnostic',
            'runtime_cpus': list(range(32, 48)), 'scope': self.plan['scope'], 'plan_sha256': self.fields['plan_SHA'],
            'fixture_plan_sha256': self.fields['fixture_plan_SHA'], 'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in self.cases},
            'runner_files_sha256': code_hashes(), 'providers': self.plan['providers'], 'start_epoch': 200,
            'drain_evidence_path': 'unit_NOT_real', 'drain_evidence_sha256': '0' * 64,
            'expected_core_loaded_stack_sha256': {path: '0' * 64 for path in ('libtorch_xpu', 'libsycl', 'libur_adapter', 'libze_')},
            'paired12288_terminal_sha256': paired_binding()['terminal_sha256'], 'proof_timing_process_separation': True,
            'cpu_READY_sha256': '0' * 64}

    def validate(self, value):
        validate_release(value, self.plan, self.fields, self.cases, code_hashes(), 100, '6', list(range(32, 48)), True)

    def trace(self, reset=False):
        name = 'actual-copy' if reset else 'FusedQKNormRopeKernel'
        return {'traceEvents': [{'ph': 'X', 'name': REPLAY_MARKER, 'ts': 100, 'dur': 200}] + [
            {'ph': 'X', 'cat': 'gpu_memcpy' if reset else 'kernel', 'name': name, 'ts': 101 + index, 'dur': 1} for index in range(64)]}

    def dot(self, reset=False):
        return '\n'.join(f'node{index} [label="zeCommandListAppendMemoryCopy" ];' if reset else
            f'node{index} [label="zeCommandListAppendLaunchKernelWithArguments" ];' for index in range(64))

    def test_all19_three_BAAB(self):
        self.assertEqual((len(self.cases), ORDER, SAMPLES, REPEATS), (19, ('port', 'source', 'source', 'port'), 21, 32))
        entries = phase_plan(self.cases)
        self.assertEqual(len(entries), 12)
        for round_index in range(3):
            rows = entries[round_index * 4:(round_index + 1) * 4]
            self.assertEqual(tuple(row['role'] for row in rows), ORDER)
            for row in rows:
                self.assertEqual(set(row['cases']), {case['name'] for case in self.cases})
        self.assertEqual(sum(case['group'] == 'actual-config-semantic-synthetic' for case in self.cases), 17)

    def test_all21_counterbalanced(self):
        for round_index in range(3):
            for slot in range(4):
                for case_index in range(19):
                    orders = [method_order(round_index, slot, case_index, sample) for sample in range(21)]
                    for method in METHODS:
                        self.assertEqual(sum(order[0] == method for order in orders), 7)

    def test_original_threshold_math_plan_functions_unchanged(self):
        for filename, names in {'runner.py': ('scope', 'validate_output'), 'protocol.py': ('load_plan', 'phase_plan', 'method_order', 'summarize', 'derived', 'graph_proof')}.items():
            original = ast.parse((self.old / filename).read_text())
            current = ast.parse((SLICE / filename).read_text())
            for name in names:
                self.assertEqual(ast.dump(function(original, name)), ast.dump(function(current, name)))

    def test_old_inclusive_event_branches_unchanged(self):
        original = ast.parse((self.old / 'runner.py').read_text())
        def event_branch(tree):
            return next(node for node in ast.walk(function(tree, 'phase')) if isinstance(node, ast.If)
                and 'starts[logical_call].record()' in ast.unparse(node))
        self.assertEqual(ast.dump(event_branch(original)), ast.dump(event_branch(self.runner)))

    def test_schema_only_not_authorization(self):
        self.validate(self.schema())
        self.assertFalse(list(SLICE.glob('*grant*.json')))

    def test_reject_protocol_mutations(self):
        for key, value in (('version', 'qk-controlled-method-diagnostic-v1'), ('gpu', 4), ('runtime_cpus', list(range(16))),
            ('proof_timing_process_separation', False), ('paired12288_terminal_sha256', '0' * 64), ('start_epoch', 100),
            ('all_fixture_sha256', {}), ('providers', {}), ('runner_files_sha256', {}), ('eight_card_concurrency_claim', True), ('cpu_READY_sha256', '')):
            with self.subTest(key=key):
                changed = self.schema()
                changed[key] = value
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_actual_paired12288_binding_limited(self):
        self.assertTrue(paired_binding()['only12288_NOT_all19'])
        for role in ('source', 'port'):
            dot = (PAIRED / 'run' / role / 'capture.dot').read_text()
            self.assertEqual(dot_structure(dot, METHODS[1])['actual_WithArguments_kernel_nodes'], 64)

    def test_window_order_capture_instantiate_sentinel_replay(self):
        proof = function(self.graphs, 'proof_graph')
        context = next(node for node in ast.walk(proof) if isinstance(node, ast.With) and ast.unparse(node.items[0].context_expr) == 'profiler')
        text = ast.unparse(context)
        self.assertLess(text.index('capture_body('), text.index('graph_metadata('))
        self.assertLess(text.index('graph_metadata('), text.index('sentinel('))
        self.assertLess(text.index('sentinel('), text.index('graph.replay()'))
        whole = ast.unparse(proof)
        self.assertLess(whole.index('output_proof('), whole.index('replay_proof('))
        self.assertLess(whole.index('graph.reset()'), whole.index("'profiler_exited_proofgraph_reset_synced'"))

    def test_timing_graph_no_profiler_constructor_or_trace(self):
        for name in ('timing_graph', 'capture_body', 'graph_metadata', 'sentinel', 'output_proof'):
            text = ast.unparse(function(self.graphs, name))
            self.assertNotIn('torch.profiler.', text)
            self.assertNotIn('record_function', text)
            self.assertNotIn('export_chrome_trace', text)
        self.assertIn('profiler_state(torch)', ast.unparse(function(self.graphs, 'timing_graph')))

    def test_fresh_proof_process_drain_before_timing(self):
        text = ast.unparse(function(self.runner, 'suite'))
        self.assertLess(text.index('proof_arguments'), text.index('child.wait()'))
        self.assertLess(text.index("f'/proc/{child.pid}'"), text.index("prefix = root / entry['phase']"))
        isolation = ast.unparse(function(self.graphs, 'isolated_timing'))
        self.assertIn('pid != os.getpid()', isolation)
        self.assertIn("Path(f'/proc/{pid}').exists()", isolation)

    def test_fake_CPU_labels_rejected(self):
        trace = self.trace()
        for event in trace['traceEvents'][1:]:
            event['cat'] = 'xpu_driver'
        with self.assertRaises(ValueError):
            replay_proof(self.dot(), trace, METHODS[1])

    def test_capture_events_cannot_replace_replay(self):
        trace = self.trace()
        for event in trace['traceEvents'][1:]:
            event['ts'] = 1
        with self.assertRaises(ValueError):
            replay_proof(self.dot(), trace, METHODS[1])

    def test_duplicate_or_absent_marker_rejected(self):
        for duplicate in (True, False):
            trace = self.trace()
            if duplicate:
                trace['traceEvents'].append(copy.deepcopy(trace['traceEvents'][0]))
            else:
                trace['traceEvents'].pop(0)
            with self.assertRaises(ValueError):
                replay_proof(self.dot(), trace, METHODS[1])

    def test_strict_G_and_R_schema(self):
        self.assertEqual(replay_proof(self.dot(), self.trace(), METHODS[1])['actual_replay_norm_kernels'], 64)
        self.assertEqual(replay_proof(self.dot(True), self.trace(True), METHODS[2])['actual_replay_norm_kernels'], 0)
        with self.assertRaises(ValueError):
            replay_proof(self.dot(), self.trace(), METHODS[2])

    def test_profiler_guard_rejects_active(self):
        fake = SimpleNamespace(autograd=SimpleNamespace(_profiler_enabled=lambda: True,
            profiler=SimpleNamespace(_is_profiler_enabled=False)))
        with self.assertRaises(ValueError):
            profiler_state(fake)

    def test_profiler_guard_requires_actual_API(self):
        with self.assertRaises(ValueError):
            profiler_state(SimpleNamespace(autograd=SimpleNamespace()))

    def test_tracing_environment_rejected(self):
        fake = SimpleNamespace(autograd=SimpleNamespace(_profiler_enabled=lambda: False,
            profiler=SimpleNamespace(_is_profiler_enabled=False)))
        with patch.dict(os.environ, {'ZET_ENABLE_API_TRACING_EXP': '1'}):
            with self.assertRaises(ValueError):
                profiler_state(fake)

    def test_proof_process_never_creates_events(self):
        text = ast.unparse(function(self.runner, 'phase'))
        self.assertLess(text.index('if proof_only:\n            del graphs'), text.index('torch.xpu.Event('))
        self.assertIn('continue', text[:text.index('torch.xpu.Event(')])
        proof = ast.unparse(function(self.graphs, 'proof_graph'))
        self.assertNotIn('torch.xpu.Event', proof)

    def test_graph_event_interval_only_original_replay(self):
        text = ast.unparse(function(self.runner, 'phase'))
        self.assertIn('starts[0].record()\n                    graphs[method].replay()\n                    ends[0].record()', text)

    def test_diagnostic_difference_unchanged(self):
        self.assertTrue(derived([1.0] * 21, [2.0] * 21)['negative_deltas_preserved'])
        self.assertTrue(derived([1.0] * 21, [2.0] * 21)['diagnostic_only_not_production_kernel_time'])

    def test_no_gpu_without_new_grant(self):
        result = subprocess.run([sys.executable, str(SLICE / 'runner.py'), '--mode', 'suite'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('New external main release REQUIRED', result.stderr)

    def test_B8_scope_first(self):
        result = subprocess.run([sys.executable, str(SLICE / 'runner.py'), '--mode', 'suite'],
            env={**os.environ, 'BATCH_SIZE': '9'}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('B8 only before GPU', result.stderr)

    def test_launcher_without_grant(self):
        result = subprocess.run(['bash', str(SLICE / 'launch.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_firstfail_no_retry(self):
        text = ast.unparse(function(self.runner, 'suite'))
        self.assertNotIn('while ', text)
        self.assertIn('no timing/retry', text)
        self.assertIn('preserve failure/no fallback/no retry', text)

    def test_no_fake_registration(self):
        for name in ('runner.py', 'graphs.py'):
            text = (SLICE / name).read_text()
            for prohibited in ('load_library(', 'torch.library.', 'register_fake(', 'torch.compile('):
                self.assertNotIn(prohibited, text)


if __name__ == '__main__':
    unittest.main(verbosity=2)

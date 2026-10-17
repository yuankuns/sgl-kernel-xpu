import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from contract import MARKER, ROOT, ROLES, VERSION, case, hashes, replay_trace, strict, unchanged, validate


class Contracts(unittest.TestCase):
    def schema(self):
        plan, fields, fixture = case()
        return {'version': VERSION, 'main_authorized': True, 'accuracy_trace_diagnostic_only': True, 'performance_authorized': False,
            'gpu': 6, 'runtime_cpus': list(range(32, 48)), 'roles': list(ROLES), 'calls_per_graph': 32,
            'case': fixture, 'providers': plan['providers'], 'driver_files_sha256': hashes(), 'core_loaded_stack_sha256': {'unit': '0' * 64},
            'drain_sha256': '0' * 64, 'old_owned_drain_sha256': '0' * 64,
            'strict_original_device_kernel_trace_gate_preserved': True, 'start_epoch': 200}

    def accept(self, value):
        with patch('contract.time.time', return_value=100):
            validate(value, True)

    def trace(self, start=110):
        events = [{'ph': 'X', 'name': MARKER, 'ts': 100, 'dur': 100, 'cat': 'user_annotation'}]
        for index in range(32):
            events.extend([{'ph': 'X', 'name': 'at::native::xpu::qk_norm_rope_kernel_xe35::Kernel_xe35', 'cat': 'kernel', 'ts': start, 'dur': 1},
                {'ph': 'X', 'name': 'reset', 'cat': 'gpu_memcpy', 'ts': start, 'dur': 1}])
        return {'traceEvents': events}

    def dot(self):
        return '\n'.join(f'K{index} [label="kernel"];' for index in range(32)) + '\n' + '\n'.join(f'C{index} [label="memcpy"];' for index in range(32))

    def test_schema_not_real_authorization(self):
        self.accept(self.schema())
        self.assertFalse((ROOT / 'main-paired-trace-grant.json').exists())

    def test_old_ready_code_retained(self):
        unchanged()

    def test_exact_original_fixture(self):
        fixture = case()[2]
        self.assertEqual(fixture['name'], 'prefill-wave-3req-12288')
        self.assertEqual(fixture['case']['tokens'], 12288)
        self.assertEqual(fixture['parameters']['tail_heads'], 144)

    def test_reject_performance(self):
        value = self.schema(); value['performance_authorized'] = True
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_old_grant_version(self):
        value = self.schema(); value['version'] = 'qk-controlled-method-diagnostic-v1'
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_GPU_change(self):
        value = self.schema(); value['gpu'] = 4
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_CPU_change(self):
        value = self.schema(); value['runtime_cpus'] = list(range(16))
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_role_selection_or_repeat(self):
        for roles in (['source'], ['port', 'source'], ['source', 'port', 'source']):
            value = self.schema(); value['roles'] = roles
            with self.assertRaises(ValueError): self.accept(value)

    def test_reject_call_count(self):
        value = self.schema(); value['calls_per_graph'] = 16
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_fixture_SHA_change(self):
        value = copy.deepcopy(self.schema()); value['case']['fixture']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_driver_or_provider_change(self):
        for name in ('driver_files_sha256', 'providers'):
            value = self.schema(); value[name] = {}
            with self.assertRaises(ValueError): self.accept(value)

    def test_reject_missing_drains(self):
        value = self.schema(); value['old_owned_drain_sha256'] = ''
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_trace_relaxation(self):
        value = self.schema(); value['strict_original_device_kernel_trace_gate_preserved'] = False
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_eightcard_claim(self):
        value = self.schema(); value['eight_card_claim'] = True
        with self.assertRaises(ValueError): self.accept(value)

    def test_reject_epoch(self):
        value = self.schema(); value['start_epoch'] = 100
        with self.assertRaises(ValueError): self.accept(value)

    def test_driver_only_trace_cannot_pass(self):
        trace = self.trace()
        for event in trace['traceEvents'][1:]: event['cat'] = 'xpu_driver'
        with self.assertRaises(ValueError): strict(self.dot(), trace)

    def test_capture_timestamps_cannot_replace_replay(self):
        with self.assertRaises(ValueError): strict(self.dot(), self.trace(start=10))

    def test_replay_device_timestamps_real_schema(self):
        result = strict(self.dot(), self.trace())
        self.assertEqual(result['actual_replay_norm_kernels'], 32)
        self.assertTrue(result['capture_events_NOT_substituted_for_replay'])

    def test_marker_unique_and_positive(self):
        trace = self.trace(); trace['traceEvents'].append(trace['traceEvents'][0])
        with self.assertRaises(ValueError): replay_trace(trace)

    def test_no_performance_timer_or_fake_registration(self):
        tree = ast.parse((ROOT / 'diagnose.py').read_text())
        attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        for name in ('Event', 'elapsed_time', 'load_library', 'register_fake', 'compile'):
            self.assertNotIn(name, attrs)

    def test_profiler_encloses_capture_instantiate_replay(self):
        tree = ast.parse((ROOT / 'diagnose.py').read_text())
        phase = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'phase')
        context = next(node for node in ast.walk(phase) if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Name) and item.context_expr.id == 'profiler' for item in node.items))
        attrs = [node.attr for node in ast.walk(context) if isinstance(node, ast.Attribute)]
        self.assertIn('instantiate', attrs); self.assertIn('replay', attrs); self.assertIn('graph', attrs)

    def test_sentinel_does_not_change_initial(self):
        tree = ast.parse((ROOT / 'diagnose.py').read_text())
        fill = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fill_']
        self.assertEqual(len(fill), 1); self.assertEqual(ast.unparse(fill[0].func.value), 'qkv')

    def test_B8_preflight_before_Torch(self):
        env = dict(os.environ, SERVING_BATCH_SIZE='9', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, str(ROOT / 'diagnose.py'), '--mode', 'suite'], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertIn('must be 8', result.stderr)

    def test_missing_grant_and_timing_selection_rejected(self):
        for args in (['--mode', 'suite'], ['--mode', 'suite', '--samples', '1'], ['--mode', 'phase', '--case', 'other']):
            result = subprocess.run([sys.executable, str(ROOT / 'diagnose.py'), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from protocol import METHODS, REPEATS, SAMPLES, SLICE, VERSION, code_hashes, derived, graph_proof, load_plan, method_order, phase_plan, summarize, validate_release


class CPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.fields, cls.cases = load_plan()
        cls.hashes = code_hashes()

    def schema(self):
        return {'version': VERSION, 'main_authorized': True, 'release_id': 'CPU_unit_schema_NOT_actual_grant', 'gpu': 6,
            'stable_gpu_exclusive_owner': 'QK controlled diagnostic', 'runtime_cpu_owner': 'QK controlled diagnostic',
            'runtime_cpus': list(range(16)), 'scope': self.plan['scope'], 'plan_sha256': self.fields['plan_SHA'],
            'fixture_plan_sha256': self.fields['fixture_plan_SHA'],
            'all_fixture_sha256': {case['name']: case['fixture']['sha256'] for case in self.cases},
            'runner_files_sha256': self.hashes, 'providers': self.plan['providers'], 'start_epoch': 200.0,
            'drain_evidence_path': 'CPU-unit-only-no-file', 'drain_evidence_sha256': '0' * 64,
            'expected_core_loaded_stack_sha256': {name: '0' * 64 for name in ('/unit/libtorch_xpu.so', '/unit/libsycl.so', '/unit/libur_adapter.so', '/unit/libze_loader.so')}}

    def validate(self, value, initial=True, now=100.0, mask='6', affinity=None):
        validate_release(value, self.plan, self.fields, self.cases, self.hashes, now, mask, list(range(16)) if affinity is None else affinity, initial)

    def test_full19_three_BAAB_no_case_selection(self):
        entries = phase_plan(self.cases)
        self.assertEqual(len(entries), 12)
        for round_index in range(3):
            rows = entries[round_index * 4:(round_index + 1) * 4]
            self.assertEqual([row['role'] for row in rows], ['port', 'source', 'source', 'port'])
            for row in rows:
                self.assertEqual(len(row['cases']), 19)
                self.assertEqual(set(row['cases']), {case['name'] for case in self.cases})

    def test_counterbalanced_each_sample_all_methods(self):
        for round_index in range(3):
            for slot in range(4):
                for case_index in range(19):
                    orders = [method_order(round_index, slot, case_index, sample) for sample in range(SAMPLES)]
                    for order in orders:
                        self.assertEqual(set(order), set(METHODS))
                    for method in METHODS:
                        self.assertEqual(sum(order[0] == method for order in orders), 7)

    def test_all_fixture_groups_and_params_unchanged(self):
        self.assertEqual(sum(case['group'] == 'actual-config-semantic-synthetic' for case in self.cases), 17)
        self.assertEqual(sum(case['group'] == 'legacy-aggregate-representative' for case in self.cases), 2)
        for case in self.cases:
            self.assertEqual(case['parameters']['num_heads_q'], 128)
            self.assertEqual(case['parameters']['num_heads_k'], 16)
            self.assertEqual(case['parameters']['tail_heads'], 144)
            self.assertEqual(case['parameters']['norm_eps'], 1e-5)

    def test_schema_validation_is_not_real_authorization(self):
        self.validate(self.schema())
        self.assertFalse(list(SLICE.glob('*grant*.json')))

    def test_old_lease_and_old_grant_rejected(self):
        for update in ({'version': 'normal-v4-model-config-perf-v1'}, {'stable_gpu_exclusive_owner': 'Mill W4'}, {'main_authorized': False}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.validate({**self.schema(), **update})

    def test_reserved_or_accuracy_gpu_rejected(self):
        for gpu in (0, 1, 2, 3, 8, 9, 10, 11):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                self.validate({**self.schema(), 'gpu': gpu}, mask=str(gpu))

    def test_mask_mismatch(self):
        with self.assertRaises(ValueError):
            self.validate(self.schema(), mask='4')

    def test_cpu_assignment_and_owner_required(self):
        with self.assertRaises(ValueError):
            self.validate(self.schema(), affinity=list(range(32, 48)))
        with self.assertRaises(ValueError):
            self.validate({**self.schema(), 'runtime_cpu_owner': 'W4'})

    def test_future_epoch_and_phase_epoch_gate(self):
        with self.assertRaises(ValueError):
            self.validate({**self.schema(), 'start_epoch': 99})
        with self.assertRaises(ValueError):
            self.validate(self.schema(), initial=False)
        self.validate(self.schema(), initial=False, now=201)

    def test_missing_drain_and_core_stack_rejected(self):
        for update in ({'drain_evidence_sha256': ''}, {'drain_evidence_path': ''}, {'expected_core_loaded_stack_sha256': {}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.validate({**self.schema(), **update})

    def test_new_code_and_package_exact_bindings(self):
        for field in ('runner_files_sha256', 'providers', 'all_fixture_sha256'):
            value = copy.deepcopy(self.schema())
            value[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(value)

    def test_no_eightcard_or_unsafe_run_id(self):
        for update in ({'release_id': '../old'}, {'eight_card_concurrency_claim': True}):
            with self.assertRaises(ValueError):
                self.validate({**self.schema(), **update})

    def test_exact21_raw_and_MAD(self):
        result = summarize(list(range(1, SAMPLES + 1)))
        self.assertEqual(result['median_us'], 11)
        self.assertEqual(result['MAD_us'], 5)
        self.assertEqual(result['raw_us'], list(range(1, SAMPLES + 1)))

    def test_no_invalid_or_filtered_samples(self):
        for values in ([1.0] * 20, [1.0] * 20 + [0], [1.0] * 20 + [-1], [1.0] * 20 + [float('nan')], [1.0] * 20 + [float('inf')]):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_delta_keeps_negative_not_kernel_time(self):
        result = derived([1.0] * SAMPLES, [2.0] * SAMPLES)
        self.assertEqual(result['raw_index_aligned_delta_us'], [-1.0] * SAMPLES)
        self.assertEqual(result['difference_of_method_medians_us'], -1)
        self.assertTrue(result['diagnostic_only_not_production_kernel_time'])

    def graph_schema(self, norm):
        nodes = [f'copy{index} [label="Memory copy"];' for index in range(REPEATS)]
        events = [{'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Device copy', 'dur': 1} for index in range(REPEATS)]
        if norm:
            nodes += [f'kernel{index} [label="Kernel"];' for index in range(REPEATS)]
            events += [{'ph': 'X', 'cat': 'kernel', 'name': 'at::native::xpu::FusedQKNormRopeKernel', 'dur': 1} for index in range(REPEATS)]
        return '\n'.join(nodes), {'traceEvents': events}

    def test_graph_schema_parser_contract_only(self):
        dot, trace = self.graph_schema(True)
        proof = graph_proof(dot, trace, METHODS[1])
        self.assertEqual(proof['actual_replay_norm_kernels'], REPEATS)
        dot, trace = self.graph_schema(False)
        self.assertEqual(graph_proof(dot, trace, METHODS[2])['actual_replay_norm_kernels'], 0)

    def test_empty_graph_CPU_labels_only_rejected(self):
        dot, trace = self.graph_schema(True)
        with self.assertRaises(ValueError):
            graph_proof('', trace, METHODS[1])
        with self.assertRaises(ValueError):
            graph_proof(dot, {'traceEvents': [{'ph': 'X', 'cat': 'cpu_op', 'name': 'FusedQKNormRopeKernel', 'dur': 1}] * REPEATS}, METHODS[1])

    def test_reset_graph_cannot_contain_norm(self):
        dot, trace = self.graph_schema(True)
        with self.assertRaises(ValueError):
            graph_proof(dot, trace, METHODS[2])

    def test_duplicate_nodes_not_fake_count(self):
        dot, trace = self.graph_schema(False)
        with self.assertRaises(ValueError):
            graph_proof('\n'.join(['same [label="Memory copy"];'] * REPEATS), trace, METHODS[2])

    def test_native_kernel_proof_and_reset_nodes_required(self):
        dot, trace = self.graph_schema(False)
        with self.assertRaises(ValueError):
            graph_proof(dot, trace, METHODS[1])

    def test_cpu_import_no_Torch_GPU_registration(self):
        import runner

        self.assertNotIn('torch', sys.modules)
        forbidden = ('load_library', 'register_fake', 'compile', 'create_thread')
        for name in ('runner.py', 'protocol.py', 'analyze.py'):
            tree = ast.parse((SLICE / name).read_text())
            calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertFalse(set(calls) & set(forbidden))

    def test_capture_reset_inside_each_logical_call(self):
        tree = ast.parse((SLICE / 'runner.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'captured')
        context = next(node for node in function.body if isinstance(node, ast.With))
        loop = context.body[0]
        self.assertIsInstance(loop, ast.For)
        expressions = [ast.unparse(node) for node in loop.body]
        self.assertLess(next(index for index, text in enumerate(expressions) if 'qkv.copy_' in text), next(index for index, text in enumerate(expressions) if 'actual_call()' in text))
        self.assertIn('is_current_stream_capturing', expressions[0])

    def test_reject_other_batch_before_Torch(self):
        import runner

        with patch.dict(os.environ, {'BATCH_SIZE': '9'}), self.assertRaises(ValueError):
            runner.scope()
        self.assertNotIn('torch', sys.modules)

    def test_missing_GPU_grant_cannot_enter_runtime(self):
        result = subprocess.run([sys.executable, str(SLICE / 'runner.py'), '--mode', 'suite'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('New external main release REQUIRED', result.stderr)

    def test_timing_and_case_selection_options_rejected(self):
        for option in ('--samples', '--case'):
            result = subprocess.run([sys.executable, str(SLICE / 'runner.py'), '--mode', 'cpu-check', option, '1'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

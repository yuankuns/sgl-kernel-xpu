import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ART / 'v5-metadata-bridge-preparation'))
import precision_v5 as checker
import prepare_and_check as guards


class PreparationTests(unittest.TestCase):
    def test_no_grant_fails_before_heavy_import(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'precision_v5.py'), '--guard-only'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('fresh main GPU10', result.stderr)

    def test_non_B8_override_rejected_before_package_check(self):
        environment = dict(os.environ, SERVING_BATCH='4')
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'precision_v5.py'), '--cpu-only'],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn('SERVING_BATCH rejected', result.stderr)

    def test_generic_Python_API_and_all_other_source_unchanged(self):
        previous = json.loads((ART / 'canonical-source-freeze-v4-icpx.json').read_text())
        current = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
        changed = [name for name in current['files'] if current['files'][name] != previous['files'][name]]
        self.assertEqual(changed, ['src/sycl/xe35/fmha/entry.cpp'])
        self.assertEqual(current['files']['python/sgl_kernel/flash_attn.py'], previous['files']['python/sgl_kernel/flash_attn.py'])

    def test_actual_model51_metadata_guards(self):
        plan = json.loads((ART / 'model-config51-v1/plan.json').read_text())
        entry = (ART / 'v5-metadata-bridge-preparation/source-snapshot-v5/src/sycl/xe35/fmha/entry.cpp').read_text()
        count = 0
        for case in plan['cases']:
            fixture = plan['fixture_files'][case['bundle']]['tensors']
            normalized = dict(case, paged_KV_pool_shape=fixture['key']['shape'],
                page_table_shape=case.get('page_table_shape', fixture['table']['shape']))
            result = guards.evaluate(entry, guards.metadata(normalized, 0), True)
            self.assertTrue(result['optimized'])
            self.assertEqual(result['private_maxK'], case['source_wrapper_inferred_max_seqlen_k'])
            self.assertEqual(case['serving_batch'], 8)
            count += 1
        self.assertEqual(count, 51)

    def test_original38_and_advanced_generic_fallback_arguments(self):
        old_entry = (ART / 'v5-metadata-bridge-preparation/before-entry.cpp').read_text()
        entry = (ART / 'v5-metadata-bridge-preparation/source-snapshot-v5/src/sycl/xe35/fmha/entry.cpp').read_text()
        cases = json.loads((ART / 'normal-v4-gpu5-accuracy-only-v1.json').read_text())['cases']
        for case in cases:
            normalized = dict(case, paged_KV_pool_shape=[520, 64, 16, 64],
                page_table_shape=[case['requests'], (case['cache_length'] + 63) // 64])
            values = guards.metadata(normalized, case['cache_length'])
            self.assertEqual(guards.evaluate(old_entry, values, False), guards.evaluate(entry, values, True))
            for name in ('softmax_lse', 'q_v_', 'kv_batch_idx_', 'leftpad_k_', 'rotary_cos_', 'rotary_sin_',
                         'seqlens_rotary_', 'scheduler_metadata_', 'rel_bias_', 'bidirectional_block_ids_',
                         'sinks_', 'q_descale_', 'k_descale_', 'v_descale_'):
                unsupported = copy.deepcopy(values)
                unsupported[name] = guards.MetadataTensor([1])
                result = guards.evaluate(entry, unsupported, True)
                self.assertFalse(result['optimized'])
                self.assertEqual(result['fallback_original_maxK'], case['cache_length'])

    def test_old_or_wrong_grant_rejected(self):
        grant = {'phase': checker.PHASE, 'gpu': 5, 'runtime_cpus': '16-31'}
        with tempfile.TemporaryDirectory(prefix='CPU-rejection-fixture-', dir=ROOT) as directory:
            path = Path(directory) / 'not-a-runtime-grant.json'
            path.write_text(json.dumps(grant))
            with self.assertRaises(AssertionError):
                checker.validate_grant(path)
        old = ART / 'model-config51-v1/gpu5-accuracy-only-barrier-v1.json'
        with self.assertRaises(AssertionError):
            checker.validate_grant(old)

    def test_no_timing_and_no_dispatch_metadata_device_read(self):
        tree = ast.parse((ROOT / 'precision_v5.py').read_text())
        observer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'native_route_observer')
        calls = [node.func.attr for node in ast.walk(observer) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse({'cpu', 'item', 'tolist', 'numpy', 'clone'} & set(calls))
        all_calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse({'Event', 'XPUGraph', 'perf_counter', 'capture_begin', 'replay'} & set(all_calls))

    def test_no_Torch_import_in_CPU_contract_suite(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()

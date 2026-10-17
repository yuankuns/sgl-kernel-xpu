import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import performance_driver as driver


class PerformancePlanTests(unittest.TestCase):
    def test_no_grant_no_GPU_import(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'performance_driver.py'),
            '--role', 'port', '--output', str(ROOT / 'never-created.json')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('fresh main GPU5', result.stderr)
        self.assertFalse((ROOT / 'never-created.json').exists())
        self.assertNotIn('torch', sys.modules)

    def test_actual_BAAB_and_three_round_contract(self):
        self.assertEqual(list(driver.SLOTS), ['B0', 'A1', 'A2', 'B3'])
        self.assertEqual(driver.ROLES['source-port'], {'before': 'source', 'after': 'port'})
        self.assertEqual(driver.ROLES['preopt-source'], {'before': 'preopt', 'after': 'source'})
        tree = ast.parse((ROOT / 'performance_driver.py').read_text())
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'debug_dump' for node in ast.walk(tree)))
        self.assertIn("graph_path.suffix == '.dot'", (ROOT / 'performance_driver.py').read_text())

    def test_serving_override_rejected_before_grant_and_GPU(self):
        environment = dict(os.environ, SERVING_BATCH='7')
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'performance_driver.py'),
            '--role', 'port', '--output', str(ROOT / 'never-created.json')],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('SERVING_BATCH rejected before torch', result.stderr)
        self.assertFalse((ROOT / 'never-created.json').exists())

    def test_offline_analyzer_rehashes_and_rejects_identity_drift(self):
        text = (ROOT / 'analyze_three_BAAB.py').read_text()
        ast.parse(text)
        self.assertIn('assert current_identity == identity', text)
        self.assertIn("digest(host_path(case['saved_output_file']))", text)
        self.assertIn('math.isfinite(sample)', text)
        self.assertIn('within-role whole-output drift', text)

    def test_real_finalseal_and_baseline_boundaries(self):
        seal = json.loads((ROOT / 'GPU10_V5_PRECISION_FINAL_SEAL.json').read_text())
        self.assertEqual(seal['raw_sha256'], '90fbd175fbb525799e77df8e943714c7d5f3c4b20ad9c3d8b4c4c54cd19d1643')
        self.assertEqual(seal['model51']['private_prefill'], 6)
        self.assertEqual(seal['model51']['private_decode'], 45)
        self.assertEqual(seal['total_checked_elements'], 1884094464)
        baseline = json.loads((ROOT / 'ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json').read_text())['preopt']
        self.assertTrue(baseline['genuine_three_tested_native_icpx_current_SHA_verified'])
        self.assertFalse(baseline['complete_original_package_all_icpx'])
        self.assertFalse(baseline['actual_model51_preopt_accuracy_proof'])
        self.assertEqual(baseline['original38_fullrefs'], 38)


if __name__ == '__main__':
    unittest.main()

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import forensic_runner as forensic


ROOT = Path(__file__).resolve().parent


class ForensicCPUContractTests(unittest.TestCase):
    def test_no_grant_no_Torch_no_GPU_no_output(self):
        output = ROOT / 'never-created.json'
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'forensic_runner.py'), '--role', 'port',
            '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('new main forensic grant required before Torch/GPU', result.stderr)
        self.assertFalse(output.exists())
        self.assertNotIn('torch', sys.modules)

    def test_B8_scope_before_grant(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'forensic_runner.py'), '--role', 'source',
            '--output', str(ROOT / 'never-created.json')], env=dict(os.environ, SERVING_BATCH='7'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('SERVING_BATCH rejected before torch', result.stderr)

    def test_capture8_two_generations_all8_reference(self):
        self.assertEqual((forensic.CAPTURE_COUNT, forensic.GENERATIONS), (8, 2))
        text = (ROOT / 'forensic_runner.py').read_text()
        self.assertIn('for output_index, captured_output in enumerate(capture_outputs)', text)
        self.assertIn('reference = full_reference(case, (*tensors, captured_output), torch)', text)
        self.assertIn('capture_outputs.append', text)
        self.assertIn('forensic-only strong references', text)

    def test_no_timing_events_warmup_graph8_not_lowered(self):
        tree = ast.parse((ROOT / 'forensic_runner.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('Event', 'elapsed_time'))
        self.assertEqual(forensic.CAPTURE_COUNT, 8)

    def test_fsynced_stage_markers_are_real_host_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'unit-test-marker.jsonl'
            with contextlib.redirect_stdout(io.StringIO()):
                journal = forensic.Journal(path)
                journal.emit('CPU_contract_unit_fixture', output_address=123)
                journal.stream.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]['sequence'], 1)
            self.assertEqual(records[0]['stage'], 'CPU_contract_unit_fixture')

    def test_real_queue_API_no_fabricated_queue(self):
        text = (ROOT / 'forensic_runner.py').read_text()
        self.assertIn('stream.sycl_queue', text)
        self.assertIn('stream._as_parameter_.value == pointer', text)
        self.assertIn('stream = torch.xpu.current_stream(device=device)', text)

    def test_capture_contains_no_GPU_readbacks_or_sync(self):
        tree = ast.parse((ROOT / 'forensic_runner.py').read_text())
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == 'graph' for item in node.items):
                found += 1
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        self.assertNotIn(child.func.attr, ('cpu', 'item', 'synchronize', 'clone'))
        self.assertEqual(found, 2)

    def test_protocol_has_lifecycle_boundaries_and_firstfail(self):
        text = (ROOT / 'forensic_runner.py').read_text()
        for marker in ('before_capture_end', 'capture_end_returned', 'replay_begin', 'replay_returned',
                       'cleanup_begin', 'cleanup_returned', 'actual_fwd_enter', 'actual_fwd_return'):
            self.assertIn(marker, text)
        controller = (ROOT / 'pair_controller.py').read_text()
        self.assertIn('automatic_retry=False', controller)
        self.assertIn("for role in ('source', 'port')", controller)


if __name__ == '__main__':
    unittest.main()

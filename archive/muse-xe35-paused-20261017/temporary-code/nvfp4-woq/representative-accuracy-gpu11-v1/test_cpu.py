import ast
import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import common


class Contracts(unittest.TestCase):
    def test_b8_negative_before_torch(self):
        for key in ('BATCH', 'BATCHES', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE'):
            with patch.dict(os.environ, {key: '9'}):
                with self.assertRaises(RuntimeError):
                    common.preflight()

    def test_no_timing_graph_or_warmup(self):
        tree = ast.parse((common.DIRECTORY / 'runner.py').read_text())
        names = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        for forbidden in ('Event', 'XPUGraph', 'elapsed_time', 'perf_counter', 'timeit_samples'):
            self.assertNotIn(forbidden, names)
        self.assertNotIn('warmup', (common.DIRECTORY / 'runner.py').read_text())
        self.assertEqual(common.SHAPES, ((2, 128, 128, 256), (2, 128, 256, 512), (4, 256, 256, 512), (2, 512, 512, 512)))

    def test_fixture_all_rows_dtype(self):
        import torch
        self.assertFalse(torch.xpu.is_initialized())
        ready = json.loads((common.DIRECTORY / 'FIXTURE_READY.json').read_text())
        for record, shape in zip(ready['cases'], common.SHAPES, strict=True):
            fixture = torch.load(common.DIRECTORY / record['fixture'], weights_only=True)
            self.assertEqual(tuple(fixture['expected'].shape), (shape[1], shape[2]))
            self.assertEqual(fixture['expected'].dtype, torch.float32)
            self.assertEqual(fixture['c'].dtype, torch.bfloat16)
            self.assertEqual(fixture['sf'].dtype, torch.uint8)
            self.assertTrue(record['full_rows_reference'])
            self.assertEqual(record['rtol'], 0.005)
            for name in (*common.INPUT_NAMES, 'c', 'sf', 'expected'):
                self.assertEqual(common.evidence(fixture[name]), record['all_tensor_identity'][name])
        self.assertFalse(torch.xpu.is_initialized())

    def test_disk_bounded(self):
        self.assertLess(common.DISK_CAP + common.DISK_HEADROOM, 10 * common.GIB)
        payload = 8 * 64 * 1024 ** 2
        for experts, rows, channels, inner in common.SHAPES:
            payload += 3 * (rows * inner + rows * inner // 32 + experts * channels * inner * 2
                            + experts * channels * inner // 8 + rows * channels * 8 + experts * 32)
        self.assertLess(payload, common.GIB)


if __name__ == '__main__':
    common.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    common.save(common.DIRECTORY / 'CPU_CONTRACT.json', {'actual_exit': 0, 'tests': result.testsRun,
                'GPU_initialized': False, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
                'disk_cap_GiB': common.DISK_CAP / common.GIB, 'extra_headroom_GiB': 1})

import json
import os
import unittest

from prepare_cpu import DIRECTORY, disk, policy
from monitor import CAP, FREE_FLOOR, MAX_SECONDS, guard


class Contract(unittest.TestCase):
    def test_original_frozen_ready(self):
        self.assertEqual(policy.verify_ready()['actual_exit'], 0)

    def test_disk_budget(self):
        budget = disk()
        self.assertLess(budget['complete_conservative_bound_GiB'], 64)
        self.assertEqual(budget['production_baseline_payload_GiB'], 23.50390625)
        self.assertEqual(CAP, 64 * 2**30)
        self.assertEqual(FREE_FLOOR, 100 * 2**30)
        self.assertEqual(MAX_SECONDS, 12 * 3600)

    def test_guard_negative_cases(self):
        metrics = {'largest_compiler_rss_gib': 0, 'largest_owned_process_rss_gib': 1,
                   'tree_rss_gib': 1, 'available_gib': 200}
        files = {'owned_artifact_bytes': 1, 'global_free_bytes': FREE_FLOOR + 1}
        self.assertIsNone(guard(metrics, files, 1))
        for changes in ({'largest_owned_process_rss_gib': 16.01}, {'tree_rss_gib': 24.01}, {'available_gib': 95.99}):
            self.assertIsNotNone(guard(metrics | changes, files, 1))
        self.assertIsNotNone(guard(metrics, files | {'owned_artifact_bytes': CAP + 1}, 1))
        self.assertIsNotNone(guard(metrics, files | {'global_free_bytes': FREE_FLOOR - 1}, 1))
        self.assertIsNotNone(guard(metrics, files, MAX_SECONDS + 1))


if __name__ == '__main__':
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    if not result.wasSuccessful():
        raise SystemExit(1)
    policy.save(DIRECTORY / 'CPU_CONTRACT.json', {'actual_exit': 0, 'tests': result.testsRun,
                'GPU_started': False, 'compiler_started': False, 'frozen_v1_modified': False})

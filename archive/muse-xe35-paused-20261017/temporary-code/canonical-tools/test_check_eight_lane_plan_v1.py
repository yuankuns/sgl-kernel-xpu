import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from check_eight_lane_plan_v1 import ARTIFACTS, MODULES, WORKLOAD, check_plan


class EightLanePlanningContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        witness = self.root / 'synthetic-planning-witness.txt'
        witness.write_text('Synthetic CPU planning fixture; not GPU evidence.\n')
        record = {'path': witness.name, 'sha256': hashlib.sha256(witness.read_bytes()).hexdigest()}
        self.plan = {'workload': dict(WORKLOAD), 'compilation_cpu_ceiling': 128,
                     'native_compilation_during_runtime': False, 'lanes': []}
        for index in range(8):
            self.plan['lanes'].append({'gpu': index, 'mode': 'accuracy',
                'runtime_cpus': list(range(index * 16, (index + 1) * 16)), 'modules': []})
        for index, module in enumerate(sorted(MODULES)):
            self.plan['lanes'][index % 8]['modules'].append({'name': module,
                'artifacts': {kind: dict(record) for kind in ARTIFACTS}})

    def rejected(self, plan):
        with self.assertRaises(ValueError):
            check_plan(plan, self.root)

    def test_planning_success_is_not_runtime_acceptance(self):
        result = check_plan(self.plan, self.root)
        self.assertEqual(result['module_count'], 11)
        self.assertEqual(result['runtime_cpu_count'], 128)
        self.assertFalse(result['GPU_work_started'])
        self.assertFalse(result['eight_card_overlap_proven'])
        self.assertFalse(result['precision_or_performance_accepted'])

    def test_nonproduction_workload_rejected(self):
        for batch in (1, 4, 16):
            plan = copy.deepcopy(self.plan)
            plan['workload']['serving_batch'] = batch
            self.rejected(plan)

    def test_duplicate_gpu_rejected(self):
        self.plan['lanes'][1]['gpu'] = 0
        self.rejected(self.plan)

    def test_early_gpu_performance_rejected(self):
        self.plan['lanes'][0].update(gpu=8, mode='performance')
        self.rejected(self.plan)

    def test_early_gpu_accuracy_eligible(self):
        self.plan['lanes'][0]['gpu'] = 8
        self.assertEqual(check_plan(self.plan, self.root)['physical_gpus'][0], 1)

    def test_cpu_overlap_rejected(self):
        self.plan['lanes'][1]['runtime_cpus'][0] = 0
        self.rejected(self.plan)

    def test_cpu_outside_budget_rejected(self):
        self.plan['lanes'][0]['runtime_cpus'][0] = 128
        self.rejected(self.plan)

    def test_missing_module_rejected(self):
        self.plan['lanes'][0]['modules'].pop()
        self.rejected(self.plan)

    def test_missing_lane_rejected(self):
        self.plan['lanes'].pop()
        self.rejected(self.plan)

    def test_artifact_changes_rejected(self):
        (self.root / 'synthetic-planning-witness.txt').write_text('changed\n')
        self.rejected(self.plan)

    def test_compilation_during_runtime_rejected(self):
        self.plan['native_compilation_during_runtime'] = True
        self.rejected(self.plan)


if __name__ == '__main__':
    unittest.main()

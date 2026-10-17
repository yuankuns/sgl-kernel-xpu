import hashlib
import json
import os
from pathlib import Path
import unittest


DIRECTORY = Path(__file__).resolve().parent


class ProposalContract(unittest.TestCase):
    def test_actual_probe(self):
        probe = json.loads((DIRECTORY / 'INSTALLED_API_CPU_PROBE.json').read_text())
        self.assertEqual(probe['actual_exit'], 0)
        self.assertEqual(probe['CPU_affinity'], list(range(80, 96)))
        for key in ('Torch_XPU_initialized', 'GPU_started', 'compiler_started',
                    'runtime_trace_available_or_working_claimed',
                    'native_argument_scalar_values_observed'):
            self.assertIs(probe[key], False)
        for name in ('instantiate', 'replay', 'raw_xpu_graph', 'raw_xpu_graph_exec'):
            self.assertIs(probe['methods'][name]['exists'], True)
        self.assertIn('replay', probe['methods']['instantiate']['doc'])
        self.assertIn("doesn't provide APIs", probe['methods']['raw_xpu_graph']['doc'])

    def test_limits_and_nonclaims(self):
        text = (DIRECTORY / 'PROPOSAL_CN.md').read_text()
        for token in ('65partial不能接受', '没有证明此版本trace能工作',
                      '独立instrumentation grant', '若工具不暴露参数则明确未观测',
                      '引用捕获本身不等于失效', '新GPU/trace/compile/grant数量均为0',
                      '首次失败先保全', 'main选择一个分支', 'mapped填成', '自动instantiate'):
            self.assertIn(token, text)


if __name__ == '__main__':
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProposalContract))
    if not result.wasSuccessful():
        raise SystemExit(1)
    seals = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (DIRECTORY / 'PROPOSAL_CN.md', DIRECTORY / 'probe_cpu.py',
                          DIRECTORY / 'INSTALLED_API_CPU_PROBE.json', Path(__file__))}
    report = {'actual_exit': 0, 'tests_run': result.testsRun,
              'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'GPU_started': False, 'compiler_started': False,
              'root_cause_proven': False, 'new_grant_created': False, 'sha256': seals}
    with (DIRECTORY / 'CPU_CONTRACT.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')

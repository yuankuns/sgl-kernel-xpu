import argparse
import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import analyzer


def sample_case(value=10):
    elapsed = value * 128 / 1000
    rows = [{'sample_index': index, 'graph_ms': elapsed, 'per_logical_call_us': elapsed * 1000 / 128, 'calls': 128} for index in range(101)]
    actual = rows[0]['per_logical_call_us']
    return {'calls': 128, 'samples': 101, 'warmup': 5, 'accuracy': 'PASS', 'raw_us': [actual] * 101, 'raw_graph_ms': [elapsed] * 101, 'p50_us': actual, 'mad_us': 0}, rows


class ContractTests(unittest.TestCase):
    def test_case_scope(self):
        self.assertEqual(sum(len(analyzer.names(component, 'semantic')) for component in analyzer.COMPONENTS), 61)
        self.assertEqual(sum(len(analyzer.names(component, 'representative')) for component in analyzer.COMPONENTS), 14)
        for component in analyzer.COMPONENTS:
            for family in analyzer.FAMILIES:
                names = analyzer.names(component, family)
                self.assertEqual(len(set(names)), len(names))

    def test_duplicate_json_key(self):
        with self.assertRaises(analyzer.Rejected):
            analyzer.decode('{"calls":128,"calls":1}')

    def test_nonfinite_json(self):
        for token in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(token=token), self.assertRaises(analyzer.Rejected):
                analyzer.decode('{"raw":' + token + '}')

    def test_valid_sample_units(self):
        result, rows = sample_case()
        analyzer.validate_sample_arrays(result, rows)

    def test_missing_sample(self):
        result, rows = sample_case()
        with self.assertRaises(analyzer.Rejected):
            analyzer.validate_sample_arrays(result, rows[:-1])

    def test_duplicate_sample(self):
        result, rows = sample_case()
        rows[1] = rows[0].copy()
        with self.assertRaises(analyzer.Rejected):
            analyzer.validate_sample_arrays(result, rows)

    def test_units_and_array_tamper(self):
        for mutation in ('units', 'raw', 'calls', 'warm', 'sample_count', 'median', 'mad'):
            result, rows = sample_case()
            if mutation == 'units':
                rows[0]['per_logical_call_us'] *= 128
            elif mutation == 'raw':
                result['raw_us'][0] += 1
            elif mutation == 'calls':
                rows[0]['calls'] = True
            elif mutation == 'warm':
                result['warmup'] = 4
            elif mutation == 'sample_count':
                result['samples'] = 100
            elif mutation == 'median':
                result['p50_us'] = 0
            else:
                result['mad_us'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(analyzer.Rejected):
                analyzer.validate_sample_arrays(result, rows)

    def test_bad_durations(self):
        for value in (0, -1, math.nan, math.inf, True, '1'):
            result, rows = sample_case()
            rows[0]['graph_ms'] = value
            with self.subTest(value=value), self.assertRaises(analyzer.Rejected):
                analyzer.validate_sample_arrays(result, rows)

    def test_case_coverage(self):
        names = analyzer.names('memory', 'semantic')
        valid = [{'case': name} for name in names]
        analyzer.validate_manifest_case_set(valid, names)
        for records in (valid[:-1], valid + [valid[0]], [valid[0]] * len(valid), list(reversed(valid))):
            with self.assertRaises(analyzer.Rejected):
                analyzer.validate_manifest_case_set(records, names)

    def test_maintenance(self):
        analyzer.validate_phase_watch({'external_processes_modified': False, 'contamination': []})
        for record in ({'external_processes_modified': False, 'contamination': [{'active': {'123': 'compiler'}}]}, {'external_processes_modified': False}, {'external_processes_modified': True, 'contamination': []}):
            with self.assertRaises(analyzer.Rejected):
                analyzer.validate_phase_watch(record)

    def test_missing_and_failed_exits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / 'status.exit'
            with self.assertRaises(analyzer.Rejected):
                analyzer.Evidence().exit0(path)
            for value in ('1', '-15', '', 'unknown'):
                path.write_text(value)
                with self.assertRaises(analyzer.Rejected):
                    analyzer.Evidence().exit0(path)

    def test_missing_controller_terminal(self):
        with self.assertRaises(analyzer.Rejected):
            analyzer.terminal_requirements(analyzer.Evidence(), analyzer.ROOT / analyzer.RUN_NAME, None)

    def test_sha_change_and_final_seal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / 'artifact.json'
            path.write_text('{}')
            evidence = analyzer.Evidence()
            evidence.file(path)
            with self.assertRaises(analyzer.Rejected):
                evidence.file(path, '0' * 64)
            path.write_text('{"changed":true}')
            with self.assertRaises(analyzer.Rejected):
                evidence.stable()

    def test_fake_or_zero_dot(self):
        for text in ('digraph "graph" {}', 'digraph "graph" { node0 [label="kernel"]; }', 'graph "graph" {}', 'digraph "graph" { L0_S0_C0 [label="fake"]; }'):
            with self.subTest(text=text), self.assertRaises(analyzer.Rejected):
                analyzer.parse_native_dot(text)

    def test_three_round_statistics_and_hold(self):
        phases = [sample_case(value)[0] for value in [10, 12, 12, 10] * 3]
        report = analyzer.statistics_case(phases)
        self.assertTrue(report['hold'])
        self.assertTrue(report['stable_regression'])
        self.assertTrue(report['regression_over5pct'])
        self.assertFalse(report['automatic_acceptance'])
        self.assertEqual(report['role_pooled_samples'], {'before': 606, 'after': 606})
        self.assertEqual(report['outliers_dropped'], 0)
        close_phases = [sample_case(value)[0] for value in [10, 10, 10, 10] * 3]
        self.assertFalse(analyzer.statistics_case(close_phases)['hold'])
        drift = [sample_case(value)[0] for value in [10, 10, 10, 12] * 3]
        self.assertTrue(analyzer.statistics_case(drift)['drift_hold'])
        cross_round = [sample_case(value)[0] for value in [10] * 4 + [12] * 4 + [10] * 4]
        self.assertTrue(analyzer.statistics_case(cross_round)['drift_hold'])
        with self.assertRaises(analyzer.Rejected):
            analyzer.statistics_case(phases[:-1])

    def test_real_Torch_CPU_wholebits(self):
        torch = analyzer.cpu_environment()
        first = {'inputs': {'ids': torch.arange(8)}, 'outputs': {'value': torch.tensor([0.0, 1.0])}}
        same = copy.deepcopy(first)
        counters = {'tensors': 0, 'bytes': 0}
        analyzer.tensor_identity(torch, first, same, 'CPU_contract', counters)
        self.assertEqual(counters['tensors'], 2)
        same['outputs']['value'][0] = -0.0
        with self.assertRaises(analyzer.Rejected):
            analyzer.tensor_identity(torch, first, same, 'CPU_signed_zero', counters)
        same = copy.deepcopy(first)
        same['inputs']['ids'][0] = 1
        with self.assertRaises(analyzer.Rejected):
            analyzer.tensor_identity(torch, first, same, 'CPU_input_identity', counters)
        same = copy.deepcopy(first)
        same['outputs']['value'] = same['outputs']['value'].double()
        with self.assertRaises(analyzer.Rejected):
            analyzer.tensor_identity(torch, first, same, 'CPU_dtype', counters)
        self.assertFalse(torch.xpu.is_initialized())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyzer.cpu_environment()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {'actual_exit': 0 if result.wasSuccessful() else 1, 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'Torch_XPU_initialized': False, 'GPU_started': False, 'compiler_started': False, 'synthetic_JSON_is_contract_only_not_hardware_evidence': True, 'analyzer_SHA256': analyzer.digest(analyzer.__file__), 'test_SHA256': analyzer.digest(__file__)}
    analyzer.save_new(args.output, evidence)
    raise SystemExit(evidence['actual_exit'])

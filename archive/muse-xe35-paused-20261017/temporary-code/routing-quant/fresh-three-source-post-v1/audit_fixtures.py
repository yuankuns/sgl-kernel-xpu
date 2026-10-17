import hashlib
import json
import os
from pathlib import Path
import time

import torch

from input_forensics import tensor_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    evidence = directory.parent / 'fresh-three-source-diagnostic-v1'
    assert not torch.xpu.is_initialized()
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    runtime_pid = int((evidence / 'container-runtime.pid').read_text())
    assert not Path(f'/proc/{runtime_pid}').exists()
    before = json.loads((evidence / 'normal-source-precheck.json').read_text())
    after = json.loads((directory / 'package-post-fresh.json').read_text())
    for field in ['package', 'Torch_provider', 'Torch_version', 'mapped_native_sha256', 'package_files_sha256', 'provenance_sha256']:
        assert before[field] == after[field], field
    assert after['GPU_initialized'] is False
    results = json.loads((evidence / 'independent-four-component-accuracy.json').read_text())
    references = json.loads((evidence / 'CPU-original-fixtures.json').read_text())['fixtures']
    assert len(results) == len(references) == 3
    cases = []
    all_passed = True
    for reference, result in zip(references, results, strict=True):
        assert reference['node_suffix'] == result['node_suffix']
        fixture_path = evidence / Path(result['fixture_file']).name
        actual = torch.load(fixture_path, weights_only=True, map_location='cpu')
        expected = torch.load(evidence / reference['fixture_file'], weights_only=True, map_location='cpu')
        assert set(actual) == set(result['tensor_SHA256'])
        for name, tensor in actual.items():
            assert tensor.device.type == 'cpu'
            assert tensor_sha256(tensor) == result['tensor_SHA256'][name], name
        for name, tensor in expected.items():
            assert tensor_sha256(tensor) == reference['tensors'][name]
            torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)
        torch.testing.assert_close(actual['actual_input'], actual['original_input'], rtol=0, atol=0)
        gates = {}
        for component in ['separate_q', 'separate_s', 'fused_q', 'fused_s']:
            expected_component = actual['expected_s' if component.endswith('_s') else 'expected_q']
            different = int(torch.count_nonzero(actual[component] != expected_component))
            try:
                torch.testing.assert_close(actual[component], expected_component, rtol=0, atol=0)
                passed = True
            except AssertionError:
                passed = False
            recorded = result['four_independent_original_zero_gates'][component]
            assert recorded['rtol'] == recorded['atol'] == 0
            assert passed == recorded['passed'] and different == recorded['different_elements']
            assert expected_component.numel() == recorded['checked_elements']
            gates[component] = {'passed': passed, 'different_elements': different, 'checked_elements': expected_component.numel()}
            all_passed &= passed
        activation_difference = int(torch.count_nonzero(actual['separate_activation'].view(torch.int16) != actual['original_CPU_BF16_activation'].view(torch.int16)))
        assert activation_difference == result['separate_activation_BF16_different_elements']
        cases.append({'node': result['node_suffix'], 'complete_fixture_file_SHA256': digest(fixture_path),
                      'tensor_SHA256': result['tensor_SHA256'], 'four_zero_gates': gates,
                      'separate_activation_BF16_different_elements': activation_difference})
    report = {'CPU_evidence_complete': True, 'accuracy_gate_passed': all_passed,
              'verified_epoch': time.time(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'GPU_initialized': torch.xpu.is_initialized(), 'container_runtime_PID_absent': runtime_pid,
              'normal_SOURCE_package_unchanged': True, 'cases': cases,
              'original_pytest_fused_assertions': 'not claimed reached; independent complete-fixture gates are separate evidence',
              'fresh_SOURCE_scope': 'three selected natives only; shared TripleOps and registration reused unchanged',
              'no_GPU_or_compile_launched': True}
    (directory / 'full-fixture-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

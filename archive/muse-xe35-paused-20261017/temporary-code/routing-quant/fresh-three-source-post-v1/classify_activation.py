import hashlib
import json
import os
from pathlib import Path
import runpy

import torch


def main():
    directory = Path(__file__).resolve().parent
    evidence = directory.parent / 'fresh-three-source-diagnostic-v1'
    assert not torch.xpu.is_initialized()
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests/test_per_token_group_quant_mxfp4.py')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == '4656ddb0d3161582c5d0a2015cdf8bd5b2b563e8653e929b7286b7b1e1cd1aab'
    quantize = runpy.run_path(str(source))['quantize_to_mxfp4']
    cases = []
    for result in json.loads((evidence / 'independent-four-component-accuracy.json').read_text()):
        tensors = torch.load(evidence / Path(result['fixture_file']).name, weights_only=True, map_location='cpu')
        reference_q, reference_s = quantize(tensors['separate_activation'])
        gates = {}
        for component, reference in [('separate_q', reference_q), ('separate_s', reference_s)]:
            different = int(torch.count_nonzero(tensors[component] != reference))
            gates[component] = {'different_elements': different, 'checked_elements': reference.numel(), 'passed': different == 0}
        coordinates = torch.nonzero(tensors['separate_activation'].view(torch.int16) != tensors['original_CPU_BF16_activation'].view(torch.int16)).tolist()
        cases.append({'node': result['node_suffix'], 'separate_quant_against_original_CPU_reference_of_actual_separate_activation': gates,
                      'all_activation_mismatch_coordinates': coordinates,
                      'first_activation_mismatch': {'index': coordinates[0],
                          'separate_value': float(tensors['separate_activation'][tuple(coordinates[0])]),
                          'original_CPU_BF16_value': float(tensors['original_CPU_BF16_activation'][tuple(coordinates[0])])} if coordinates else None})
    assert not torch.xpu.is_initialized()
    report = {'CPU_evidence_complete': True, 'GPU_initialized': False, 'original_reference_file_SHA256': source_sha,
              'no_monkeypatch_or_reference_change': True, 'cases': cases,
              'scope': 'CPU-only classification of saved tensors; does not change failed original accuracy gate or prove exact internal arithmetic mechanism'}
    (directory / 'activation-boundary-classification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value for name, value in report.items() if name != 'cases'}))
    for case in cases:
        print(json.dumps({name: value for name, value in case.items() if name != 'all_activation_mismatch_coordinates'}))


if __name__ == '__main__':
    main()

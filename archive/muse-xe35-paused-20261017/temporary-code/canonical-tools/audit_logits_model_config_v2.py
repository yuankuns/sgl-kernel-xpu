import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    import torch

    assert not torch.xpu.is_initialized()
    reports, outputs = {}, {}
    raw = {}
    for label in ('before', 'after'):
        assert (args.evidence / f'{label}.exit').read_text().strip() == '0'
        report_path = args.evidence / f'{label}.json'
        tensor_path = report_path.with_suffix('.pt')
        reports[label] = json.loads(report_path.read_text())
        outputs[label] = torch.load(tensor_path, map_location='cpu', weights_only=True)
        raw[label] = {'report_sha256': digest(report_path), 'tensor_sha256': digest(tensor_path)}
        assert reports[label]['label'] == label and reports[label]['phase'] == 'accuracy'
        assert reports[label]['gpu'] == '11' and reports[label]['samples'] == reports[label]['calls'] == 0
        assert len(reports[label]['results']) == 6
        assert all(case['accuracy'] == 'PASS' for case in reports[label]['results'])
        assert reports[label]['serving_batch'] == 8
        assert (reports[label]['isl'], reports[label]['osl'], reports[label]['tp'], reports[label]['ep']) == (4096, 16, 1, 1)
    for key in ('weight_sha256', 'model_config_sha256', 'driver_sha256'):
        assert reports['before'][key] == reports['after'][key]
    expected_cases = {f'{operation}-{shape}' for operation in ('embedding', 'mask')
                      for shape in ('decode8', 'prefill-wave12288', 'prefill-wave20480')}
    assert set(outputs['before']) == set(outputs['after']) == expected_cases
    comparisons = {}
    for case in sorted(expected_cases):
        original, port = outputs['before'][case], outputs['after'][case]
        assert original.keys() == port.keys()
        comparisons[case] = {}
        for key in original:
            before, after = original[key], port[key]
            assert before.device.type == after.device.type == 'cpu'
            assert before.dtype == after.dtype and before.shape == after.shape
            differences = int((before.contiguous().view(torch.uint8) != after.contiguous().view(torch.uint8)).sum())
            assert differences == 0, (case, key, differences)
            comparisons[case][key] = {'elements': before.numel(), 'bitwise_differences': differences}
    assert not torch.xpu.is_initialized()
    report = {'actual_exit': 0, 'scope': 'six config-matching synthetic kernel cases; not raw/model replay',
              'raw': raw, 'comparisons': comparisons, 'GPU_initialized': False,
              'full_independent_references_per_provider': True,
              'model_hidden': 8192, 'serving_batch': 8,
              'weight_sha256': reports['before']['weight_sha256'],
              'model_config_sha256': reports['before']['model_config_sha256']}
    with (args.evidence / 'bitwise-audit.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'whole_tensor_pairs': sum(map(len, comparisons.values()))}))


if __name__ == '__main__':
    main()

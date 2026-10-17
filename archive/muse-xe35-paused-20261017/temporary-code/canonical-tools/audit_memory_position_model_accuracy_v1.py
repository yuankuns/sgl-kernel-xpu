import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    import torch

    assert not torch.xpu.is_initialized()
    run = ROOT / 'memory-position-model-config-accuracy-v1-run1'
    assert json.loads((run / 'execution-complete.json').read_text())['actual_exit'] == 0
    comparisons, raw = {}, {}
    for component, expected_count in (('memory', 21), ('position', 34)):
        providers, reports = {}, {}
        raw[component] = {}
        for label in ('before', 'after'):
            prefix = run / f'{component}-{label}'
            assert prefix.with_suffix('.exit').read_text().strip() == '0'
            reports[label] = json.loads(prefix.with_suffix('.json').read_text())
            providers[label] = torch.load(prefix.with_suffix('.pt'), map_location='cpu', weights_only=True)
            report = reports[label]
            assert report['component'] == component and report['label'] == label
            assert report['phase'] == 'accuracy' and report['gpu'] == '11'
            assert report['serving_batch'] == 8 and report['samples'] == report['calls'] == 0
            assert report['timing'] == 'none'
            assert (report['isl'], report['osl'], report['tp'], report['ep']) == (4096, 16, 1, 1)
            assert len(report['results']) == len(providers[label]) == expected_count
            assert all(record['accuracy'] == 'PASS' for record in report['results'])
            raw[component][label] = {'report_sha256': digest(prefix.with_suffix('.json')),
                                     'tensor_sha256': digest(prefix.with_suffix('.pt'))}
        assert reports['before']['driver_sha256'] == reports['after']['driver_sha256']
        assert reports['before']['model_config_sha256'] == reports['after']['model_config_sha256']
        assert providers['before'].keys() == providers['after'].keys()
        comparisons[component] = {}
        for case, original in providers['before'].items():
            port = providers['after'][case]
            assert original.keys() == port.keys() == {'inputs', 'outputs'}
            comparisons[component][case] = {}
            for group in ('inputs', 'outputs'):
                assert original[group].keys() == port[group].keys()
                for key, source in original[group].items():
                    after = port[group][key]
                    assert source.device.type == after.device.type == 'cpu'
                    assert source.dtype == after.dtype and source.shape == after.shape
                    differences = int((source.contiguous().view(torch.uint8) != after.contiguous().view(torch.uint8)).sum())
                    assert differences == 0, (component, case, group, key, differences)
                    comparisons[component][case][f'{group}/{key}'] = {'elements': source.numel(), 'bitwise_differences': differences}
    assert not torch.xpu.is_initialized()
    report = {'actual_exit': 0, 'original_gate_tolerance': 'integer exact zero',
              'model_shape_semantic_cases_per_provider': 55, 'raw': raw, 'comparisons': comparisons,
              'scope': 'synthetic B8 serving waves3/5 and15decode contracts; not allocator/whole-model/raw replay',
              'GPU_initialized_by_auditor': False, 'performance_verified': False}
    with (run / 'independent-bitwise-audit.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'cases_per_provider': 55,
                      'whole_tensor_pairs': sum(len(case) for component in comparisons.values() for case in component.values())}))


if __name__ == '__main__':
    main()

import json
import os

import common


def main():
    common.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    common.verify_ready()
    import torch
    assert not torch.xpu.is_initialized()
    for name in ('source', 'after', 'wrapper', 'host-supervisor'):
        assert (common.DIRECTORY / f'{name}.exit').read_text().strip() == '0', name
    release = json.loads((common.DIRECTORY / 'HOST_RELEASE.json').read_text())
    assert release['GPU11_lock_probe_exit'] == 0 and release['wrapper_PID_absent']
    fixture_record = json.loads((common.DIRECTORY / 'FIXTURE_READY.json').read_text())
    seals = {}
    checked = 0
    for index, shape in enumerate(common.SHAPES):
        fixture = torch.load(common.DIRECTORY / fixture_record['cases'][index]['fixture'], weights_only=True)
        pairs = []
        for label in ('source', 'after'):
            directory = common.DIRECTORY / 'results' / label / f'case-{index}'
            raw = torch.load(directory / 'fullraw-before-check.pt', weights_only=True)
            evidence = json.loads((directory / 'fullraw-before-check.json').read_text())
            report = json.loads((directory / 'result.json').read_text())
            assert report['actual_exit'] == 0 and report['shape'] == list(shape)
            assert report['timing_samples'] == [] and report['full_rows_reference']['checked'] == shape[1] * shape[2]
            for name, tensor in raw.items():
                assert common.evidence(tensor) == evidence[name]
            for name in (*common.INPUT_NAMES, 'c', 'sf', 'expected'):
                assert torch.equal(common.bits(raw[name]), common.bits(fixture[name]))
            torch.testing.assert_close(raw['actual'], fixture['expected'], rtol=0.005, atol=0.005)
            assert raw['actual'].dtype == torch.float32
            initial = torch.load(directory / 'all-inputs-before.pt', weights_only=True)
            assert initial['workspace_initial'].dtype == torch.uint8
            assert initial['workspace_initial'].numel() == 64 * 1024 * 1024
            assert torch.count_nonzero(initial['workspace_initial']).item() == 0
            for name in common.INPUT_NAMES:
                assert torch.equal(common.bits(initial[name]), common.bits(fixture[name]))
            for path in directory.iterdir():
                if path.is_file():
                    seals[str(path.relative_to(common.DIRECTORY))] = common.digest(path)
            pairs.append(raw)
            checked += shape[1] * shape[2]
        for name in (*common.INPUT_NAMES, 'c', 'sf', 'actual'):
            assert torch.equal(common.bits(pairs[0][name]), common.bits(pairs[1][name])), name
    for label in ('source', 'after'):
        assert json.loads((common.DIRECTORY / f'{label}.drain.json').read_text())['all_owned_drained']
    assert not torch.xpu.is_initialized()
    common.save(common.DIRECTORY / 'INDEPENDENT_CPU_POSTCHECK.json', {'actual_exit': 0,
                'cases': 4, 'phases': 2, 'independent_CPU_FP32_elements_checked': checked,
                'whole_SOURCE_input_prep_output_bits': True, 'GPU_initialized': False,
                'GPU11_lock_probe_exit': 0, 'all_known_owned_descendants_drained': True,
                'raw_sha256': seals, 'performance_accepted': False})
    print(json.dumps({'actual_exit': 0, 'full_reference_elements_two_phases': checked,
                      'whole_SOURCE_bits': True, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

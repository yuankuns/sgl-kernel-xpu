import argparse
import json
import os
from pathlib import Path
import sys

import common


def fixtures():
    import torch
    assert not torch.xpu.is_initialized()
    sys.path.insert(0, str(common.SOURCE / 'benchmark'))
    factory = common.module('original_representative_factory', common.SOURCE / 'benchmark/bench_nvfp4_woq_group_gemm.py')
    reference = common.module('original_accuracy_reference', common.SOURCE / 'benchmark/accuracy_reference.py')
    scale = common.module('original_scale_reference', common.SOURCE / 'tests/nvfp4_utils.py')
    torch.set_num_threads(8)
    directory = common.DIRECTORY / 'fixtures'
    directory.mkdir()
    records = []
    for index, (experts, rows, channels, inner) in enumerate(common.SHAPES):
        values = factory.build_inputs(experts, rows // experts, channels, inner, 'cpu')
        assert values[-1] == rows
        tensors = dict(zip(common.INPUT_NAMES, values[:-1]))
        tensors = {name: value.contiguous().clone() for name, value in tensors.items()}
        multipliers, exponents = scale.nvfp4_woq_prep_ref(tensors['b_sf'], tensors['b_gs'])
        expected = torch.empty((rows, channels), dtype=torch.float32)
        for expert in range(experts):
            start = expert * (rows // experts)
            end = start + rows // experts
            activations = tensors['a_q'][start:end].float()
            activations *= torch.exp2(tensors['a_s'][start:end].float() - 127).repeat_interleave(32, -1)
            weights = reference.dequantize_nvfp4_prepared(tensors['b_fp4'][expert], multipliers[expert], exponents[expert], 'mxfp8')
            expected[start:end] = activations @ weights.T
        tensors.update(c=multipliers.contiguous().clone(), sf=exponents.contiguous().clone(), expected=expected)
        path = directory / f'case-{index}.pt'
        torch.save(tensors, path)
        records.append({'shape': [experts, rows, channels, inner], 'fixture': str(path.relative_to(common.DIRECTORY)),
                        'sha256': common.digest(path), 'all_tensor_identity': {name: common.evidence(value) for name, value in tensors.items()},
                        'full_rows_reference': True, 'reference': 'original CPU FP32 independently decoded quantized operands',
                        'rtol': 0.005, 'atol': 0.005})
    common.save(common.DIRECTORY / 'FIXTURE_READY.json', {'actual_exit': 0, 'GPU_initialized': torch.xpu.is_initialized(), 'cases': records})


def seal():
    import torch
    assert not torch.xpu.is_initialized()
    for label in ('source', 'after'):
        record = json.loads((common.DIRECTORY / f'provider-{label}.json').read_text())
        assert record['GPU_initialized'] is False
        assert record['after_manifest_sha256'] == 'a4492cadb1020191435f184e2859bbe3ae47e7305a98b02e782562296a4e9dba'
        assert (common.DIRECTORY / f'provider-{label}.exit').read_text().strip() == '0'
    for name in ('fixture', 'contracts'):
        assert (common.DIRECTORY / f'{name}.exit').read_text().strip() == '0'
    prior = json.loads((common.ARTIFACT / 'normal-v5-performance-gpu1-v1/CPU_READY.json').read_text())
    assert prior['GRF512_both'] is True
    fixture = json.loads((common.DIRECTORY / 'FIXTURE_READY.json').read_text())
    assert fixture['actual_exit'] == 0 and fixture['GPU_initialized'] is False
    paths = [common.DIRECTORY / name for name in ('provider-source.json', 'provider-after.json', 'FIXTURE_READY.json', 'CPU_CONTRACT.json')]
    paths += [common.DIRECTORY / case['fixture'] for case in fixture['cases']]
    common.save(common.DIRECTORY / 'CPU_READY.json', {'actual_exit': 0, 'GPU_initialized': False, 'GRF512_both': True,
                'canonical_version': 'canonical-v5-icpx', 'dependencies_sha256': common.dependencies(),
                'evidence_sha256': {str(path.relative_to(common.DIRECTORY)): common.digest(path) for path in paths},
                'SOURCE_registration_scope': 'four fresh tested natives, preserved genuine historical other registration; not full icpx SOURCE',
                'shapes': common.SHAPES, 'independent_reference': 'CPU FP32 all rows/all outputs, original rtol=atol=0.005',
                'disk_cap_bytes': common.DISK_CAP, 'disk_headroom_bytes': common.DISK_HEADROOM,
                'new_grant_created': False, 'performance_authorized': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('source', 'after', 'fixture', 'seal'))
    args = parser.parse_args()
    common.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    if args.action in common.SITES:
        common.save(common.DIRECTORY / f'provider-{args.action}.json', common.provider(args.action))
    elif args.action == 'fixture':
        fixtures()
    else:
        seal()
    print(json.dumps({'actual_exit': 0, 'action': args.action, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

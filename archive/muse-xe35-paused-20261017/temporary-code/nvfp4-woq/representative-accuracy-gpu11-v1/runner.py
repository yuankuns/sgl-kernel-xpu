import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=('source', 'after'))
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    common.preflight()
    authorization = common.grant(args.grant)
    assert os.environ['ZE_AFFINITY_MASK'] == '11'
    assert time.time() >= authorization['start_epoch']
    common.save(common.DIRECTORY / f'runtime-provider-{args.label}.json', common.provider(args.label))
    import torch
    import sgl_kernel
    assert not torch.xpu.is_initialized()
    assert sgl_kernel.is_xe3_arch(), 'Xe3.5 only'
    sys.path.insert(0, str(common.SOURCE / 'benchmark'))
    factory = common.module('original_representative_factory', common.SOURCE / 'benchmark/bench_nvfp4_woq_group_gemm.py')
    reference = common.module('original_accuracy_reference', common.SOURCE / 'benchmark/accuracy_reference.py')
    torch.set_num_threads(8)
    suffix = '_xe35' if args.label == 'after' else ''
    prep = getattr(sgl_kernel, 'nvfp4_woq_prep' + suffix)
    gemm = getattr(sgl_kernel, 'nvfp4_woq_mxfp8_scaled_grouped_mm' + suffix)
    fixtures = json.loads((common.DIRECTORY / 'FIXTURE_READY.json').read_text())
    records = []
    for index, (experts, rows, channels, inner) in enumerate(common.SHAPES):
        torch.xpu.synchronize()
        time.sleep(2)
        directory = common.DIRECTORY / 'results' / args.label / f'case-{index}'
        directory.mkdir(parents=True)
        expected = torch.load(common.DIRECTORY / fixtures['cases'][index]['fixture'], weights_only=True)
        values = factory.build_inputs(experts, rows // experts, channels, inner, 'xpu')
        assert values[-1] == rows
        inputs = dict(zip(common.INPUT_NAMES, values[:-1]))
        workspace = torch.zeros(64 * 1024 * 1024, device='xpu', dtype=torch.uint8)
        initial = {name: value.cpu().contiguous().clone() for name, value in inputs.items()}
        initial['workspace_initial'] = workspace.cpu()
        torch.save(initial, directory / 'all-inputs-before.pt')
        identities = {name: common.evidence(value) for name, value in initial.items()}
        common.save(directory / 'input-identity.json', identities)
        for name in common.INPUT_NAMES:
            assert identities[name] == fixtures['cases'][index]['all_tensor_identity'][name], name
        multipliers, exponents = prep(inputs['b_sf'], inputs['b_gs'])
        torch.save({'c': multipliers.cpu().contiguous().clone(), 'sf': exponents.cpu().contiguous().clone(),
                    'expected': expected['expected']}, directory / 'prep-before-gemm.pt')
        actual = torch.zeros((rows, channels), device='xpu', dtype=torch.float32)
        gemm(actual, inputs['a_q'], inputs['b_fp4'], inputs['a_s'], exponents, multipliers,
             inputs['problems'], inputs['offsets'], workspace)
        torch.xpu.synchronize()
        saved = {name: value.cpu().contiguous().clone() for name, value in inputs.items()}
        saved.update(c=multipliers.cpu().contiguous().clone(), sf=exponents.cpu().contiguous().clone(),
                     actual=actual.cpu().contiguous().clone(), expected=expected['expected'])
        torch.save(saved, directory / 'fullraw-before-check.pt')
        common.save(directory / 'fullraw-before-check.json', {name: common.evidence(value) for name, value in saved.items()})
        for name in common.INPUT_NAMES:
            assert torch.equal(common.bits(saved[name]), common.bits(expected[name])), 'Input mutated: ' + name
        assert saved['c'].dtype == expected['c'].dtype
        assert saved['sf'].dtype == expected['sf'].dtype
        assert saved['actual'].dtype == torch.float32
        assert torch.equal(common.bits(saved['c']), common.bits(expected['c'])), 'Prep c'
        assert torch.equal(common.bits(saved['sf']), common.bits(expected['sf'])), 'Prep sf'
        torch.testing.assert_close(saved['actual'], expected['expected'], rtol=0.005, atol=0.005)
        report = reference.verify_grouped_output(actual, inputs['a_q'], inputs['b_fp4'], multipliers,
                    exponents, [rows // experts] * experts, 'mxfp8', inputs['a_s'], max_rows=rows // experts)
        assert report['checked'] == rows * channels
        if args.label == 'after':
            original = torch.load(common.DIRECTORY / 'results/source' / f'case-{index}' / 'fullraw-before-check.pt', weights_only=True)
            for name in (*common.INPUT_NAMES, 'c', 'sf', 'actual'):
                assert saved[name].dtype == original[name].dtype
                assert torch.equal(common.bits(saved[name]), common.bits(original[name])), 'Whole SOURCE bits: ' + name
            original_inputs = json.loads((common.DIRECTORY / 'results/source' / f'case-{index}' / 'input-identity.json').read_text())
            assert identities == original_inputs, 'All initial inputs including workspace SOURCE identity'
        records.append({'index': index, 'shape': [experts, rows, channels, inner], 'actual_exit': 0,
                        'all_input_identity': True, 'all_prep_bits': True, 'full_rows_reference': report,
                        'whole_SOURCE_bits': args.label == 'after', 'timing_samples': []})
        common.save(directory / 'result.json', records[-1])
        del workspace, values, inputs, initial, saved, expected, multipliers, exponents, actual
    common.save(common.DIRECTORY / f'{args.label}-summary.json', {'actual_exit': 0, 'cases': records,
                'performance_authorized': False, 'timing_samples': [], 'normal_AFTER': args.label == 'after'})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        common.save(common.DIRECTORY / ('failure-' + sys.argv[1] + '.json'),
                    {'actual_exit': 1, 'error': repr(error), 'traceback': traceback.format_exc(), 'retry': False})
        raise

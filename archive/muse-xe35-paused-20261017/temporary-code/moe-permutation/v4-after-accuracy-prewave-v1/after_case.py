import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from checks import ROOT, ROUND, actual_normal_import, digest, grant_gate, write_json


def parse_args():
    parser = argparse.ArgumentParser(description='Genuine MoE v4 AFTER pre-wave accuracy only; no performance/timing options')
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    args = parser.parse_args()
    if os.environ.get('BATCHES', '8') != '8' or os.environ.get('ZE_AFFINITY_MASK') != '10':
        parser.error('B8/GPU10 only; stop before Torch import or GPU work')
    return args


def reference_xe35(torch, ids_cpu, experts, alignment, upper_rows):
    tokens, topk = ids_cpu.shape
    flat_ids = ids_cpu.flatten().tolist()
    raw_counts = [flat_ids.count(expert) for expert in range(experts)]
    expected_counts = [(count + alignment - 1) // alignment * alignment for count in raw_counts]
    expected_perm_src = [0] * upper_rows
    expected_expanded_to_permuted = [0] * (tokens * topk)
    expected_scale_map = [[row, 0] for row in range(upper_rows)]
    start = 0
    for expert, padded in enumerate(expected_counts):
        indices = [index for index, value in enumerate(flat_ids) if value == expert]
        for rank, index in enumerate(indices):
            row = start + rank
            expected_perm_src[row] = index // topk
            expected_expanded_to_permuted[index] = row
            expected_scale_map[row] = [start, padded]
        for row in range(start + len(indices), start + padded):
            expected_scale_map[row] = [start, -padded]
        start += padded
    return [torch.tensor(values, dtype=torch.int32) for values in
            [expected_counts, expected_perm_src, expected_expanded_to_permuted, expected_scale_map]]


def main():
    args = parse_args()
    if (ROUND / 'after.json').exists():
        raise RuntimeError('Refusing to overwrite pre-wave evidence')
    manifest, manifest_file, site, protected, grant = grant_gate(True)
    torch, sgl_kernel, mapped = actual_normal_import(manifest, manifest_file, site, False)
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError('Actual normal v4 package must report genuine Xe35 3.5')
    if os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('Real container CPU affinity must remain96–111')
    source = json.loads((ROOT / 'source-native-icpx-v1-accuracy/fresh.json').read_text())
    if source['torch'] != torch.__version__:
        raise RuntimeError('SOURCE/AFTER Torch build differs')
    operation = torch.ops.sgl_kernel.mxfp4_small_permutation_xe35.default
    inputs_file = ROUND / 'shared-SOURCE-inputs.pt'
    if digest(inputs_file) != grant['grant']['shared_SOURCE_inputs_sha256']:
        raise RuntimeError('Shared actual-SOURCE-byte-verified input file changed')
    shared_inputs = torch.load(inputs_file, weights_only=True, map_location='cpu')
    metadata = {'version': 'canonical-v4-icpx', 'scope': vars(args), 'run_kind': 'MoE pre-wave AFTER accuracy only',
                'accuracy_only': True, 'performance_collected': False, 'timing_collected': False,
                'graphs_or_events_created': False, 'warmup_performed': False, 'formal_perf_barrier_consumed': False,
                'production_precision': 'strict original int32/int64 inputs, int32 outputs', 'atol': 0, 'rtol': 0,
                'torch': torch.__version__, 'package': sgl_kernel.__file__, 'gpu_mask': os.environ['ZE_AFFINITY_MASK'],
                'device_properties': str(torch.xpu.get_device_properties(0)),
                'actual_container_CPU_affinity': sorted(os.sched_getaffinity(0)),
                'fresh_grant': grant, 'protected_source': protected, 'cases': [], 'full_tensor_bitwise_checks': []}
    metadata.update(mapped)
    print(json.dumps(metadata), flush=True)
    for source_case in source['cases']:
        torch.xpu.synchronize()
        time.sleep(2)
        ids_cpu = shared_inputs[source_case['case']]
        if hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest() != source_case['input_sha256']:
            raise RuntimeError('Full input bytes differ from actual validated SOURCE')
        tokens, experts, topk, alignment = source_case['tokens'], source_case['experts'], source_case['topk'], source_case['alignment']
        assert tokens == 8 and experts == 256 and topk == 8 and alignment == 4
        assert ids_cpu.dtype == (torch.int32 if source_case['dtype'] == 'torch.int32' else torch.int64)
        pairs = tokens * topk
        upper_rows = pairs * alignment
        ids = ids_cpu.to('xpu')
        outputs = [torch.empty(experts, device='xpu', dtype=torch.int32),
                   torch.empty(upper_rows, device='xpu', dtype=torch.int32),
                   torch.empty(pairs, device='xpu', dtype=torch.int32),
                   torch.empty((upper_rows, 2), device='xpu', dtype=torch.int32)]
        operation(ids, *outputs, alignment)
        expected = reference_xe35(torch, ids_cpu, experts, alignment, upper_rows)
        actual = [output.cpu() for output in outputs]
        source_output_file = ROOT / 'source-native-icpx-v1-accuracy' / source_case['outputs']
        if digest(source_output_file) != source_case['outputs_sha256']:
            raise RuntimeError('Validated complete SOURCE output artifact changed')
        source_outputs = torch.load(source_output_file, weights_only=True, map_location='cpu')
        assert len(actual) == len(expected) == len(source_outputs) == 4
        for index, (output, reference, before) in enumerate(zip(actual, expected, source_outputs)):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            assert output.dtype == torch.int32 and output.shape == before.shape
            identical = torch.equal(output, before)
            metadata['full_tensor_bitwise_checks'].append({'case': source_case['case'], 'output': index,
                                                          'elements': output.numel(), 'equal': identical})
            if not identical:
                raise RuntimeError('Complete AFTER versus actual SOURCE tensor mismatch: ' + source_case['case'])
        output_path = ROUND / ('after-' + source_case['case'] + '-outputs.pt')
        torch.save(actual, output_path)
        case = {'case': source_case['case'], 'tokens': tokens, 'topk': topk, 'experts': experts, 'alignment': alignment,
                'dtype': str(ids_cpu.dtype), 'accuracy': 'strict_CPU_exact_and_SOURCE_full_bitwise_pass',
                'outputs': output_path.name, 'outputs_sha256': digest(output_path),
                'input_sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()}
        metadata['cases'].append(case)
        write_json(ROUND / 'after.json', metadata)
        print(json.dumps(case), flush=True)
    torch.xpu.synchronize()
    time.sleep(2)
    prefill_ids = torch.zeros((8 * 4096, 8), device='xpu', dtype=torch.int32)
    prefill_outputs = [torch.empty(256, device='xpu', dtype=torch.int32),
                       torch.empty(prefill_ids.numel() * 4, device='xpu', dtype=torch.int32),
                       torch.empty(prefill_ids.numel(), device='xpu', dtype=torch.int32),
                       torch.empty((prefill_ids.numel() * 4, 2), device='xpu', dtype=torch.int32)]
    try:
        operation(prefill_ids, *prefill_outputs, 4)
    except RuntimeError as error:
        assert 'invalid small permutation dimensions or alignment' in str(error)
        metadata['prefill'] = {'tokens': 32768, 'pairs': 262144, 'status': 'unsupported_source_contract',
                               'rejection_verified': True, 'error': str(error), 'performance_ms': None}
    else:
        raise AssertionError('small permutation unexpectedly accepted unsupported prefill')
    write_json(ROUND / 'after.json', metadata)
    print(json.dumps(metadata['prefill']), flush=True)
    torch.xpu.synchronize()

    assert len(metadata['cases']) == 6 and len(metadata['full_tensor_bitwise_checks']) == 24
    metadata['accuracy_passed'] = True
    metadata['full_tensor_bitwise_passed'] = True
    metadata['protected_source_post'] = grant_gate(True)[3]
    write_json(ROUND / 'after.json', metadata)


if __name__ == '__main__':
    main()

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from source_accuracy_package_check import digest, mapped_provenance, preflight


def parse_args():
    parser = argparse.ArgumentParser(description='SOURCE historical versus fresh icpx accuracy only; no timing/graphs/events')
    parser.add_argument('label', choices=['historical', 'fresh'])
    parser.add_argument('--grant-file', required=True)
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    args = parser.parse_args()
    if os.environ.get('BATCHES', '8') != '8':
        parser.error('B8 only; rejected before Torch import/GPU work')
    root = Path(__file__).resolve().parent
    gpu = '4' if root.name == 'bf16-gemm' else '10'
    if os.environ.get('ZE_AFFINITY_MASK') != gpu:
        parser.error('SOURCE-only accuracy card must be ' + gpu)
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
    root = Path(__file__).resolve().parent
    out = root / 'source-native-icpx-v1-accuracy'
    if (out / (args.label + '.json')).exists():
        raise RuntimeError('Refusing to overwrite original SOURCE accuracy evidence')
    manifest, grant = preflight(root, args.grant_file, 4 if root.name == 'bf16-gemm' else 10)
    expected_site = (root / ('source/site/sgl_kernel' if args.label == 'historical' else 'fresh-source-icpx-v1/site/sgl_kernel')).resolve()
    expected_files = manifest['SOURCE_original_files'] if args.label == 'historical' else manifest['site_files']
    import torch
    import sgl_kernel

    if Path(sgl_kernel.__file__).resolve() != expected_site / '__init__.py':
        raise RuntimeError('Must import the real original normal SOURCE package')
    mapped = mapped_provenance(expected_site, expected_files, manifest['native_SONAME'])
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError('SOURCE accuracy requires real Xe35 capability 3.5')
    metadata = {'label': args.label, 'version': 'source-native-icpx-v1', 'scope': vars(args),
                'accuracy_only': True, 'performance_acceptance_eligible': False,
                'timing_collected': False, 'graphs_or_events_created': False,
                'package': sgl_kernel.__file__, 'torch': torch.__version__,
                'device': torch.xpu.get_device_name(0), 'device_properties': str(torch.xpu.get_device_properties(0)),
                'gpu_mask': os.environ.get('ZE_AFFINITY_MASK'), 'fresh_grant': grant,
                'source_HEAD': manifest['source_HEAD'], 'source_pin': manifest['source_pin'],
                'baseline_scope': manifest['scope'], 'cases': []}
    metadata.update(mapped)
    operation = torch.ops.sgl_kernel.mxfp4_small_permutation.default
    metadata['accuracy'] = 'strict_bitwise'
    print(json.dumps(metadata), flush=True)
    for dtype in [torch.int32, torch.int64]:
        for active_experts in [23, 256, 1]:
            torch.xpu.synchronize()
            time.sleep(2)
            tokens, experts, topk, alignment = 8, 256, 8, 4
            pairs = tokens * topk
            upper_rows = pairs * alignment
            ids_cpu = (torch.arange(pairs, dtype=dtype).reshape(tokens, topk) * 7) % active_experts
            ids = ids_cpu.to('xpu')
            outputs = [torch.empty(experts, device='xpu', dtype=torch.int32),
                       torch.empty(upper_rows, device='xpu', dtype=torch.int32),
                       torch.empty(pairs, device='xpu', dtype=torch.int32),
                       torch.empty((upper_rows, 2), device='xpu', dtype=torch.int32)]

            def operation_xe35():
                operation(ids, *outputs, alignment)

            operation_xe35()
            expected = reference_xe35(torch, ids_cpu, experts, alignment, upper_rows)
            actual = [output.cpu() for output in outputs]
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
            case_name = f'{str(dtype).split(".")[-1]}-active{active_experts}'
            output_path = out / (args.label + '-' + case_name + '-outputs.pt')
            torch.save(actual, output_path)
            case = {'case': case_name, 'tokens': tokens, 'topk': topk, 'experts': experts, 'alignment': alignment,
                    'dtype': str(dtype), 'accuracy': 'bitwise_pass',
                    'outputs': output_path.name, 'outputs_sha256': digest(output_path),
                    'input_sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()}
            metadata['cases'].append(case)
            (out / (args.label + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
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
    (out / (args.label + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['prefill']), flush=True)
    torch.xpu.synchronize()


if __name__ == '__main__':
    main()

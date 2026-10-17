import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def parse_args_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=['source', 'port', 'diagnostic-port'])
    parser.add_argument('run')
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--canonical-version', choices=['canonical-v4-icpx'], default='canonical-v4-icpx')
    parser.add_argument('--accuracy-only', action='store_true')
    parser.add_argument('--performance-grant')
    parser.add_argument('--grant-file', required=True)
    args = parser.parse_args()
    if not args.run.startswith('canonical-v4-icpx-') or Path(args.run).name != args.run:
        parser.error('A new v4-icpx raw evidence prefix is required; v2 is immutable')
    if args.calls <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error('Invalid measurement sample counts')
    if os.environ.get('BATCHES', '8') != '8':
        parser.error('BATCHES must be 8; rejected before torch import or GPU work')
    if Path(args.canonical_version).name != args.canonical_version:
        parser.error('Canonical version must be a directory basename')
    if args.diagnostic or args.label == 'diagnostic-port':
        parser.error('Old diagnostic launcher is retired; official canonical package only')
    expected_gpu = '10' if args.accuracy_only else '4'
    if os.environ.get('ZE_AFFINITY_MASK') != expected_gpu:
        parser.error('GPU lease requires ZE_AFFINITY_MASK=' + expected_gpu + '; rejected before torch import')
    return args


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


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


def capture_xe35(torch, function, warmup, calls):
    for _ in range(warmup):
        function()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(calls):
            function()
    for _ in range(warmup):
        graph.replay()
    torch.xpu.synchronize()
    return graph


def measure_xe35(torch, graph, calls):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def main_xe35():
    args = parse_args_xe35()
    from checks import grant_preflight, runtime_provenance

    root = Path(__file__).resolve().parent.parent
    evidence = Path(__file__).resolve().parent
    canonical_manifest, canonical_manifest_file, canonical_site, fresh_grant = grant_preflight(root, args.grant_file, args.accuracy_only)
    import torch
    import sgl_kernel

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Source/port benchmark requires Xe35 (capability 3.5)")

    expected_site = canonical_site if args.label == 'port' else root / 'fresh-source-icpx-v1/site/sgl_kernel'
    assert Path(sgl_kernel.__file__).resolve() == expected_site / '__init__.py'
    operation = getattr(torch.ops.sgl_kernel, 'mxfp4_small_permutation' + ('_xe35' if args.label == 'port' else '')).default
    library = 'libsgl-ops-sycl-' + ('mxfp4_small_permutation' if args.label == 'port' else 'Mxfp4SmallPermutation') + '.so'
    maps = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'common_ops.' in line or library in line or 'mxfp4_small_permutation_port_payload' in line})
    assert str(expected_site / library) in maps
    assert all(str(expected_site) in name for name in maps)
    metadata = {'label': args.label, 'run': args.run, 'torch': torch.__version__, 'package': sgl_kernel.__file__,
                'verification_kind': 'port-kernel diagnostic, not official integration' if args.diagnostic else 'official integration',
                'loaded_libraries': {name: digest_xe35(Path(name)) for name in maps},
                'device': torch.xpu.get_device_name(0), 'device_properties': str(torch.xpu.get_device_properties(0)),
                'gpu_mask': os.environ.get('ZE_AFFINITY_MASK'), 'scope': vars(args), 'accuracy': 'bitwise',
                'timing': 'xpu_graph_events', 'cases': []}
    metadata.update(runtime_provenance(root, args.label, expected_site, canonical_manifest, canonical_manifest_file))
    metadata['icpx_source_proofs'] = canonical_manifest['checker_proofs']
    metadata['fresh_grant'] = fresh_grant
    metadata['actual_CPU_affinity'] = sorted(os.sched_getaffinity(0))
    assert metadata['actual_CPU_affinity'] == list(range(96, 112))
    SOURCE_inputs = torch.load(evidence / 'shared-SOURCE-inputs.pt', weights_only=True)
    metadata['timing'] = 'accuracy_only_no_timing' if args.accuracy_only else 'xpu_graph_events'
    metadata['performance_acceptance_eligible'] = not args.accuracy_only
    assert metadata['gpu_mask'] == ('10' if args.accuracy_only else '4')
    print(json.dumps(metadata), flush=True)
    for dtype in [torch.int32, torch.int64]:
        for active_experts in [23, 256, 1]:
            torch.xpu.synchronize()
            time.sleep(2)
            tokens, experts, topk, alignment = 8, 256, 8, 4
            pairs = tokens * topk
            upper_rows = pairs * alignment
            ids_cpu = (torch.arange(pairs, dtype=dtype).reshape(tokens, topk) * 7) % active_experts
            case_name = f'{str(dtype).split(".")[-1]}-active{active_experts}'
            assert torch.equal(ids_cpu, SOURCE_inputs[case_name])
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
            output_path = evidence / (args.run + '-' + case_name + '-outputs.pt')
            torch.save(actual, output_path)
            graph = None if args.accuracy_only else capture_xe35(torch, operation_xe35, args.warmup, args.calls)
            samples = []
            for index in range(0 if args.accuracy_only else args.repeat):
                sample = measure_xe35(torch, graph, args.calls)
                samples.append(sample)
                print(json.dumps({'case': case_name, 'repeat': index, 'ms': sample}), flush=True)
            case = {'case': case_name, 'tokens': tokens, 'topk': topk, 'experts': experts, 'alignment': alignment,
                    'dtype': str(dtype), 'samples_ms': {'permutation': samples},
                    'medians_ms': {'permutation': statistics.median(samples) if samples else None}, 'accuracy': 'bitwise_pass',
                    'outputs': output_path.name, 'outputs_sha256': digest_xe35(output_path),
                    'input_sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()}
            metadata['cases'].append(case)
            (evidence / (args.run + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
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
    (evidence / (args.run + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['prefill']), flush=True)
    torch.xpu.synchronize()


if __name__ == '__main__':
    main_xe35()

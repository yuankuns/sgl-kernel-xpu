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
    parser.add_argument('--calls', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--canonical-version', choices=['canonical-v3-icpx'], default='canonical-v3-icpx')
    parser.add_argument('--accuracy-only', action='store_true')
    parser.add_argument('--performance-grant')
    parser.add_argument('--grant-file', required=True)
    args = parser.parse_args()
    if not args.run.startswith('canonical-v3-icpx-') or Path(args.run).name != args.run:
        parser.error('A new v3-icpx raw evidence prefix is required; v2 is immutable')
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
    from official_package_check import grant_preflight, runtime_provenance

    root = Path(__file__).resolve().parent.parent
    canonical_manifest, canonical_manifest_file, canonical_site, fresh_grant = grant_preflight(root, args.grant_file, args.accuracy_only)
    import torch
    import sgl_kernel

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Source/port benchmark requires Xe35 (capability 3.5)")

    expected_site = canonical_site if args.label == 'port' else root / 'source/site/sgl_kernel'
    assert Path(sgl_kernel.__file__).resolve() == expected_site / '__init__.py'
    suffix = '_xe35' if args.label == 'port' else ''
    gemm = getattr(sgl_kernel, 'bf16_gemm_nt' + suffix)
    pack = getattr(sgl_kernel, 'pack_bf16_gemm_weight' + suffix)
    operation = getattr(torch.ops.sgl_kernel, 'bf16_gemm_nt' + suffix).default
    maps = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'common_ops.' in line or 'libsgl-ops-sycl-bf16_gemm_nt' in line})
    assert str(expected_site / 'libsgl-ops-sycl-bf16_gemm_nt.so') in maps
    assert all(str(expected_site) in name for name in maps)
    metadata = {'label': args.label, 'run': args.run, 'torch': torch.__version__,
                'verification_kind': 'port-kernel diagnostic, not official integration' if args.diagnostic else 'official integration',
                'package': sgl_kernel.__file__, 'loaded_libraries': {name: digest_xe35(Path(name)) for name in maps},
                'device': torch.xpu.get_device_name(0), 'device_properties': str(torch.xpu.get_device_properties(0)),
                'gpu_mask': os.environ.get('ZE_AFFINITY_MASK'), 'scope': vars(args),
                'timing': 'xpu_graph_events', 'atol': 5e-3, 'rtol': 5e-3, 'cases': []}
    metadata.update(runtime_provenance(root, args.label, expected_site, canonical_manifest, canonical_manifest_file))
    metadata['icpx_source_proofs'] = canonical_manifest['checker_proofs']
    metadata['fresh_grant'] = fresh_grant
    metadata['timing'] = 'accuracy_only_no_timing' if args.accuracy_only else 'xpu_graph_events'
    metadata['performance_acceptance_eligible'] = not args.accuracy_only
    assert metadata['gpu_mask'] == ('10' if args.accuracy_only else '4')
    print(json.dumps(metadata), flush=True)
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(42)
    names = ['native_kernel', 'packed_kernel', 'native_api', 'packed_api']
    for columns in [8192, 18432, 202048]:
        torch.xpu.synchronize()
        time.sleep(2)
        activation_cpu = torch.randn((8, 8192), dtype=torch.bfloat16, generator=generator)
        weight_cpu = torch.randn((columns, 8192), dtype=torch.bfloat16, generator=generator)
        activations = activation_cpu.to('xpu')
        weight = weight_cpu.to('xpu')
        packed_weight = pack(weight)
        torch.testing.assert_close(packed_weight[:, :columns], weight.T, atol=0, rtol=0)
        reference = activation_cpu.float() @ weight_cpu.float().T
        native_output = torch.empty((8, columns), dtype=torch.bfloat16, device='xpu')
        packed_output = torch.empty_like(native_output)

        def native_kernel_xe35():
            operation(native_output, activations, weight, 0)
            return native_output

        def packed_kernel_xe35():
            operation(packed_output, activations, packed_weight, 2)
            return packed_output

        functions = {'native_kernel': native_kernel_xe35, 'packed_kernel': packed_kernel_xe35,
                     'native_api': lambda: gemm(activations, weight),
                     'packed_api': lambda: gemm(activations, packed_weight, packed_weight=True, output_n=columns)}
        outputs = {}
        accuracy = {}
        for name in names:
            device_output = functions[name]()
            assert device_output.dtype == torch.bfloat16
            output = device_output.float().cpu()
            torch.testing.assert_close(output, reference, atol=5e-3, rtol=5e-3)
            outputs[name] = output
            error = (output - reference).abs()
            accuracy[name] = {'max_abs': error.max().item(), 'passed': True, 'output_dtype': str(device_output.dtype),
                              'max_error_over_tolerance': (error / (5e-3 + 5e-3 * reference.abs())).max().item()}
        output_path = root / f'{args.run}-N{columns}-outputs.pt'
        torch.save(outputs, output_path)
        graphs = {} if args.accuracy_only else {name: capture_xe35(torch, functions[name], args.warmup, args.calls) for name in names}
        samples = {name: [] for name in names}
        for index in range(0 if args.accuracy_only else args.repeat):
            for name in names if index % 2 == 0 else list(reversed(names)):
                sample = measure_xe35(torch, graphs[name], args.calls)
                samples[name].append(sample)
                print(json.dumps({'n': columns, 'name': name, 'repeat': index, 'ms': sample}), flush=True)
        case = {'m': 8, 'n': columns, 'k': 8192, 'dtype': 'bf16', 'accuracy': accuracy,
                'packed_shape': list(packed_weight.shape), 'samples_ms': samples,
                'medians_ms': {name: statistics.median(values) if values else None for name, values in samples.items()},
                'outputs': output_path.name, 'outputs_sha256': digest_xe35(output_path),
                'input_sha256': hashlib.sha256(activation_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'weight_sha256': hashlib.sha256(memoryview(weight_cpu.view(torch.uint8).numpy())).hexdigest()}
        metadata['cases'].append(case)
        (root / (args.run + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'n': columns, 'medians_ms': case['medians_ms'], 'accuracy': accuracy}), flush=True)
        del graphs, functions, packed_weight, weight, weight_cpu, activation_cpu, activations
    torch.xpu.synchronize()


if __name__ == '__main__':
    main_xe35()

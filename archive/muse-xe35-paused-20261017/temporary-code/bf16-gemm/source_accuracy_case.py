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
    gemm = sgl_kernel.bf16_gemm_nt
    pack = sgl_kernel.pack_bf16_gemm_weight
    operation = torch.ops.sgl_kernel.bf16_gemm_nt.default
    metadata.update({'atol': 5e-3, 'rtol': 5e-3, 'production_dtype': 'bf16'})
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
        output_path = out / f'{args.label}-N{columns}-outputs.pt'
        torch.save(outputs, output_path)
        case = {'m': 8, 'n': columns, 'k': 8192, 'dtype': 'bf16', 'accuracy': accuracy,
                'packed_shape': list(packed_weight.shape),
                'outputs': output_path.name, 'outputs_sha256': digest(output_path),
                'input_sha256': hashlib.sha256(activation_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'weight_sha256': hashlib.sha256(memoryview(weight_cpu.view(torch.uint8).numpy())).hexdigest()}
        metadata['cases'].append(case)
        (out / (args.label + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'n': columns, 'accuracy': accuracy}), flush=True)
        del functions, packed_weight, weight, weight_cpu, activation_cpu, activations
    torch.xpu.synchronize()


if __name__ == '__main__':
    main()

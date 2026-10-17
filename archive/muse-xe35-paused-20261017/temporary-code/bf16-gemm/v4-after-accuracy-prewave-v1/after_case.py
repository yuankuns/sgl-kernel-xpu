import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from checks import ROOT, ROUND, actual_normal_import, digest, grant_gate, write_json


def parse_args():
    parser = argparse.ArgumentParser(description='Genuine v4 AFTER pre-wave accuracy only; no performance/timing options')
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    args = parser.parse_args()
    if os.environ.get('BATCHES', '8') != '8' or os.environ.get('ZE_AFFINITY_MASK') != '4':
        parser.error('B8/GPU4 only; stop before Torch import or GPU work')
    return args


def main():
    args = parse_args()
    if (ROUND / 'after.json').exists():
        raise RuntimeError('Refusing to overwrite pre-wave evidence')
    manifest, manifest_file, site, protected, grant = grant_gate(True)
    torch, sgl_kernel, mapped = actual_normal_import(manifest, manifest_file, site, False)
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError('Actual normal v4 package must report genuine Xe35 3.5')
    source = json.loads((ROOT / 'source-native-icpx-v1-accuracy/fresh.json').read_text())
    if source['torch'] != torch.__version__:
        raise RuntimeError('SOURCE/AFTER Torch build differs')
    gemm = sgl_kernel.bf16_gemm_nt_xe35
    pack = sgl_kernel.pack_bf16_gemm_weight_xe35
    operation = torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default
    metadata = {'version': 'canonical-v4-icpx', 'scope': vars(args), 'run_kind': 'pre-wave AFTER accuracy only',
                'accuracy_only': True, 'performance_collected': False, 'timing_collected': False,
                'graphs_or_events_created': False, 'warmup_performed': False, 'formal_perf_barrier_consumed': False,
                'production_dtype': 'bf16', 'atol': 5e-3, 'rtol': 5e-3, 'torch': torch.__version__,
                'package': sgl_kernel.__file__, 'gpu_mask': os.environ['ZE_AFFINITY_MASK'],
                'device_properties': str(torch.xpu.get_device_properties(0)),
                'fresh_grant': grant, 'protected_source': protected, 'cases': [], 'full_tensor_bitwise_checks': []}
    metadata.update(mapped)
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
        source_case = next(case for case in source['cases'] if case['n'] == columns)
        if source_case['input_sha256'] != hashlib.sha256(activation_cpu.view(torch.uint8).numpy().tobytes()).hexdigest():
            raise RuntimeError('Production activation input differs from precision-validated SOURCE')
        if source_case['weight_sha256'] != hashlib.sha256(memoryview(weight_cpu.view(torch.uint8).numpy())).hexdigest():
            raise RuntimeError('Production weight input differs from precision-validated SOURCE')
        source_path = ROOT / 'source-native-icpx-v1-accuracy' / source_case['outputs']
        if digest(source_path) != source_case['outputs_sha256']:
            raise RuntimeError('Validated full SOURCE output tensor artifact changed')
        source_outputs = torch.load(source_path, weights_only=True, map_location='cpu')
        if set(source_outputs) != set(outputs):
            raise RuntimeError('Missing full SOURCE projection path')
        for name in names:
            identical = torch.equal(outputs[name], source_outputs[name])
            metadata['full_tensor_bitwise_checks'].append({'n': columns, 'path': name,
                                                          'elements': outputs[name].numel(), 'equal': identical})
            if not identical:
                raise RuntimeError('Full SOURCE versus genuine v4 AFTER bitwise mismatch: ' + name)
        output_path = ROUND / f'after-N{columns}-outputs.pt'
        torch.save(outputs, output_path)
        case = {'m': 8, 'n': columns, 'k': 8192, 'dtype': 'bf16', 'accuracy': accuracy,
                'packed_shape': list(packed_weight.shape),
                'outputs': output_path.name, 'outputs_sha256': digest(output_path),
                'input_sha256': hashlib.sha256(activation_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'weight_sha256': hashlib.sha256(memoryview(weight_cpu.view(torch.uint8).numpy())).hexdigest()}
        metadata['cases'].append(case)
        (ROUND / 'after.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'n': columns, 'accuracy': accuracy}), flush=True)
        del functions, packed_weight, weight, weight_cpu, activation_cpu, activations
    torch.xpu.synchronize()

    assert len(metadata['full_tensor_bitwise_checks']) == 12
    metadata['accuracy_passed'] = True
    metadata['full_tensor_bitwise_passed'] = True
    metadata['protected_source_post'] = grant_gate(True)[3]
    write_json(ROUND / 'after.json', metadata)


if __name__ == '__main__':
    main()

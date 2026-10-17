import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual, modeled):
    different = actual.view(torch.int16) != modeled.view(torch.int16)
    coordinates = different.nonzero().tolist()
    return {'mismatches': len(coordinates), 'total_elements': actual.numel(),
            'actual_SHA256': tensor_digest(actual), 'modeled_SHA256': tensor_digest(modeled),
            'all_mismatches': [{'coordinate': coordinate,
                                'actual': actual[tuple(coordinate)].item(),
                                'modeled': modeled[tuple(coordinate)].item(),
                                'actual_bits': int(actual.view(torch.int16)[tuple(coordinate)]) & 65535,
                                'modeled_bits': int(modeled.view(torch.int16)[tuple(coordinate)]) & 65535}
                               for coordinate in coordinates]}


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    original = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    after = Path('/root/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35')
    fixtures = artifact / 'fresh-three-source-diagnostic-v1'
    assert sorted(os.sched_getaffinity(0)) == list(range(32, 48))
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    historical_binary = artifact / 'historical-source-normal-v1/site/sgl_kernel/libsgl-ops-sycl-TripleOps.so'
    fresh_binary = artifact / 'fresh-source-icpx-native-v2/site/sgl_kernel/libsgl-ops-sycl-TripleOps.so'
    assert digest(historical_binary) == digest(fresh_binary) == 'e5946388c78064722885345fec961ad6bb12caa846c369eee1beccc35d19aed6'
    symbols = subprocess.check_output(['nm', '-a', '-C', str(historical_binary)], text=True)
    (directory / 'historical-TripleOps.nm-demangled.txt').write_text(symbols)
    selected = sorted(set(line for line in symbols.splitlines() if 'silu_mul_dpcpp_functor<sycl::_V1::ext::oneapi::bfloat16,' in line))
    assert selected and all('silu_mul_dpcpp_functor<sycl::_V1::ext::oneapi::bfloat16, sycl::_V1::ext::oneapi::bfloat16>' in line for line in selected)
    (directory / 'actual-BF16-silu-instantiations.txt').write_text('\n'.join(selected) + '\n')
    commit = '5fb14b2f257e16564f3c4eb5ba9db8b9579ff555'
    parent_body = subprocess.check_output(['git', '-C', str(original), 'show', f'{commit}^:src/sycl/TripleOps.cpp'])
    (directory / 'TripleOps.pre-5fb14b2.cpp').write_bytes(parent_body)
    history = subprocess.check_output(['git', '-C', str(original), 'show', commit, '--', 'src/sycl/TripleOps.cpp'], text=True)
    (directory / 'macro-accumulator-history.diff').write_text(history)
    source_paths = [original / 'src/sycl/TripleOps.cpp', after / 'src/sycl/TripleOps.cpp',
                    Path(torch.__file__).parent / 'include/ATen/OpMathType.h',
                    Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/bfloat16.hpp')]
    evidence = {}
    for index, path in enumerate(source_paths):
        snapshot = directory / f'evidence-{index}-{path.name}'
        snapshot.write_bytes(path.read_bytes())
        evidence[str(path)] = {'SHA256': digest(path), 'snapshot': str(snapshot)}
    manifest = json.loads((fixtures / 'independent-four-component-accuracy.json').read_text())
    results = []
    for record in manifest:
        path = Path(record['fixture_file'])
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        for name, expected in record['tensor_SHA256'].items():
            assert tensor_digest(tensors[name]) == expected['sha256'], name
        inputs = tensors['actual_input']
        assert torch.equal(inputs.view(torch.int16), tensors['original_input'].view(torch.int16))
        gate, up = inputs.float().chunk(2, dim=-1)
        denominator = 1.0 + torch.exp(-gate)
        silu_fp32 = gate / denominator
        fp32_final = (silu_fp32 * up).to(torch.bfloat16)
        rounded_silu = silu_fp32.to(torch.bfloat16)
        bf16_intermediate = (rounded_silu.float() * up).to(torch.bfloat16)
        actual = tensors['separate_activation']
        output = directory / (path.stem + '-CPU-models.pt')
        torch.save({'input': inputs, 'actual_native_activation': actual,
                    'original_CPU_BF16_reference': tensors['original_CPU_BF16_activation'],
                    'FP32_denominator': denominator, 'FP32_silu': silu_fp32,
                    'BF16_silu_intermediate': rounded_silu,
                    'FP32_final_BF16': fp32_final, 'BF16_intermediate_final_BF16': bf16_intermediate}, output)
        results.append({'nodeid_suffix': record['node_suffix'], 'fixture_file': str(path),
                        'fixture_file_SHA256': digest(path), 'CPU_models_file': str(output),
                        'CPU_models_file_SHA256': digest(output),
                        'all_original_tensor_hashes_reverified': True,
                        'FP32_final_vs_actual': compare(actual, fp32_final),
                        'BF16_intermediate_vs_actual': compare(actual, bf16_intermediate),
                        'original_CPU_reference_vs_actual': compare(actual, tensors['original_CPU_BF16_activation']),
                        'FP32_final_vs_original_CPU_reference': compare(tensors['original_CPU_BF16_activation'], fp32_final),
                        'first_element': {'gate': gate[0, 0].item(), 'up': up[0, 0].item(),
                                          'FP32_silu': silu_fp32[0, 0].item(),
                                          'BF16_silu': rounded_silu[0, 0].item(),
                                          'FP32_final_BF16': fp32_final[0, 0].item(),
                                          'BF16_intermediate_final_BF16': bf16_intermediate[0, 0].item(),
                                          'actual_native': actual[0, 0].item()}})
    assert not torch.xpu.is_initialized()
    report = {'epoch': time.time(), 'actual_exit_scope': 'CPU evidence collection/comparison only; does not turn original pytest failures into success',
              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'Torch_provider': torch.__file__,
              'Torch_provider_SHA256': digest(Path(torch.__file__)), 'GPU_initialized': False,
              'GPU_or_compile_work_started': False, 'quant_actual_activation_repeated': False,
              'historical_and_fresh_TripleOps_SHA256': digest(historical_binary),
              'actual_binary_BF16_silu_accumulator_type': 'sycl::ext::oneapi::bfloat16 (real ELF symbols, all selected instantiations)',
              'current_HEAD': subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip(),
              'pre_5fb14b2_TripleOps_SHA256': digest(directory / 'TripleOps.pre-5fb14b2.cpp'),
              'source_and_header_evidence': evidence,
              'arithmetic_type_path': 'old macro: T_to=SYCL BF16; at::opmath_type<T_to>=SYCL BF16; expf converts argument to float; 1.0f+expf remains FP32; BF16/float overload returns BF16, then BF16*BF16 returns BF16. Current4cb5 macro instead passes at::opmath_type<at::BFloat16>=float.',
              'provenance_limits': ['Reused historic TripleOps was not recompiled in finite fresh SOURCE; current checkout source hash is not historical binary build provenance.',
                                    'Parent source macro and real binary template types agree; this does not prove the complete historical source file or compiler flags/header hashes used to build that binary.',
                                    'Header snapshots are current SDK/Torch, not authenticated historical build headers.',
                                    'CPU torch.exp and device expf may differ in approximation/FP32 evaluation; any BF16 model residual remains unexplained, not waved away.'],
              'results': results}
    (directory / 'activation-causal-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'CPU_evidence_exit': 0, 'results': [{key: value for key, value in result.items() if key in ['nodeid_suffix', 'first_element']} | {name: result[name]['mismatches'] for name in ['FP32_final_vs_actual', 'BF16_intermediate_vs_actual', 'original_CPU_reference_vs_actual', 'FP32_final_vs_original_CPU_reference']} for result in results]}, indent=2))


if __name__ == '__main__':
    main()

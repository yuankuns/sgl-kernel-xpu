import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import time

from providers import preflight, runtime_grant, verify_provider

preflight()

import torch


def tensor_hash_xe35(tensor):
    digest = hashlib.sha256()
    for chunk in tensor.split(1, dim=0):
        digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def full_tensor_evidence(tensor, cache, name, create, exact):
    digest = hashlib.sha256()
    maximum = 0.0
    equal = True
    for chunk_index, chunk in enumerate(tensor.split(16 if name != 'output' else 1024)):
        actual = chunk.contiguous().cpu()
        digest.update(actual.view(torch.uint8).numpy().tobytes())
        path = cache / f'{name}-{chunk_index}.pt'
        if create:
            torch.save(actual, path)
        else:
            expected = torch.load(path, weights_only=True)
            if exact:
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), f'full {name} differs in chunk {chunk_index}'
            else:
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
            maximum = max(maximum, (actual.float() - expected.float()).abs().max().item())
            equal = equal and torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    return {'sha256': digest.hexdigest(), 'max_abs': maximum, 'exact': equal,
            'all_elements_checked': True, 'first_phase_cache_created': create}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=('canonical-v5-icpx',), default='canonical-v5-icpx')
    parser.add_argument('label', choices=('source', 'after'))
    parser.add_argument('--phase', type=int, required=True, choices=range(4))
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--tail-only', action='store_true')
    args = parser.parse_args()
    preflight()
    assert args.round == 0
    runtime_grant()
    provider = verify_provider(args.label, cpu_only=True)
    artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq')
    result_dir = pathlib.Path(__file__).resolve().parent / 'results' / args.run
    result_dir.mkdir(parents=True, exist_ok=True)
    import sgl_kernel

    expected_package = (artifact / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017/site/sgl_kernel' if args.label == 'source' else artifact / 'canonical-v5-icpx/site/sgl_kernel')
    assert pathlib.Path(sgl_kernel.__file__).parent == expected_package
    target = pathlib.Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
    specification = importlib.util.spec_from_file_location('independent_reference', target / 'benchmark/accuracy_reference.py')
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)
    source_tree = pathlib.Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
    specification = importlib.util.spec_from_file_location('original_scale_reference', source_tree / 'tests/nvfp4_utils.py')
    scale_reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(scale_reference)
    assert sgl_kernel.is_xe3_arch(), 'Actual architecture must be Xe3.5'
    suffix = '_xe35' if args.label == 'after' else ''
    prep = getattr(sgl_kernel, 'nvfp4_woq_prep' + suffix)
    bf16 = getattr(sgl_kernel, 'nvfp4_woq_bf16_grouped_mm' + suffix)
    mxfp8 = getattr(sgl_kernel, 'nvfp4_woq_mxfp8_scaled_grouped_mm' + suffix)
    torch.set_num_threads(8)
    print(json.dumps({'diagnostic_only': False, 'official_acceptance': False,
                      'package': sgl_kernel.__file__, 'registration': sgl_kernel.common_ops.__file__,
                      'gpu': torch.xpu.get_device_name(0), 'gpu_mask': os.environ.get('ZE_AFFINITY_MASK'),
                      'scope': 'B8/ISL4096/OSL16/TP1/EP1', 'label': args.label,
                      'round': args.round, 'phase': args.phase}), flush=True)
    shapes = [(4, 11, 192, 64), (3, 65, 256, 128)] if args.tail_only else [(256, rows, 8192, inner) for inner in (8192, 4096) for rows in (64, 262144)]
    results = []
    for experts, rows, channels, inner in shapes:
        torch.xpu.synchronize()
        time.sleep(2)
        torch.manual_seed(20261017 + rows + inner)
        counts = [1, 7, 0, 3] if experts == 4 else [rows // experts] * experts
        if experts != 4:
            for expert in torch.randperm(experts)[:rows % experts].tolist():
                counts[expert] += 1
        problems = torch.tensor([[count, channels, inner] for count in counts], device='xpu', dtype=torch.int32)
        offsets = torch.tensor([sum(counts[:expert]) for expert in range(experts)], device='xpu', dtype=torch.int32)
        weights = torch.randint(0, 256, (experts, channels, inner // 2), device='xpu', dtype=torch.uint8)
        raw_scales = torch.randint(1, 9, (experts, channels, inner // 16), device='xpu', dtype=torch.int8).to(torch.float8_e4m3fn)
        global_scales = torch.exp2(torch.arange(experts, device='xpu', dtype=torch.float32) % 5 - 2)
        multipliers, exponents = prep(raw_scales, global_scales)
        direct_reference = args.round == 0 and args.phase in (0, 1)
        if direct_reference:
            for expert in range(experts):
                expected_c, expected_sf = scale_reference.nvfp4_woq_prep_ref(raw_scales[expert:expert + 1].cpu(), global_scales[expert:expert + 1].cpu())
                assert torch.equal(multipliers[expert:expert + 1].cpu(), expected_c)
                assert torch.equal(exponents[expert:expert + 1].cpu(), expected_sf)
        cache = result_dir / f'E{experts}-M{rows}-N{channels}-K{inner}'
        cache.mkdir(exist_ok=True)
        create = args.round == 0 and args.phase == 0
        scales_report = {
            'c': full_tensor_evidence(multipliers, cache, 'c', create, True),
            'sf': full_tensor_evidence(exponents, cache, 'sf', create, True),
        }
        workspace = torch.empty(64 * 1024 * 1024, device='xpu', dtype=torch.uint8)
        for mode in (('bf16', 'mxfp8') if args.tail_only else ('bf16',)):
            torch.xpu.synchronize()
            time.sleep(2)
            activations = torch.randint(-4, 5, (rows, inner), device='xpu', dtype=torch.int8).to(torch.bfloat16 if mode == 'bf16' else torch.float8_e4m3fn)
            activation_scales = torch.full((rows, inner // 32), 127, device='xpu', dtype=torch.uint8) if mode == 'mxfp8' else None
            output = torch.empty((rows, channels), device='xpu', dtype=torch.float32)
            if mode == 'bf16':
                function = lambda: bf16(output, activations, weights, exponents, multipliers, problems, offsets, workspace)
            else:
                function = lambda: mxfp8(output, activations, weights, activation_scales, exponents, multipliers, problems, offsets, workspace)
            function()
            torch.xpu.synchronize()
            independent = reference.verify_grouped_output(output, activations, weights, multipliers, exponents, counts, mode, activation_scales, max_rows=max(counts) if args.tail_only else 3) if direct_reference else {'reference': 'full output/scales compared to independently validated first fresh icpx SOURCE phase; identical input hashes', 'direct_reference': False}
            mode_cache = cache / mode
            mode_cache.mkdir(exist_ok=True)
            output_report = full_tensor_evidence(output, mode_cache, 'output', create, True)
            inputs = {name: tensor_hash_xe35(tensor) for name, tensor in
                      (('packed', weights), ('raw_scales', raw_scales), ('globals', global_scales), ('activations', activations))}
            inputs_path = mode_cache / 'inputs.json'
            if create:
                inputs_path.write_text(json.dumps(inputs, sort_keys=True))
            else:
                assert json.loads(inputs_path.read_text()) == inputs, 'SOURCE-to-normal-AFTER inputs must be identical'
            torch.xpu.synchronize()
            time.sleep(2)
            samples = []
            assert samples == []
            report = {'label': args.label, 'shape': [experts, rows, channels, inner], 'counts': counts,
                      'mode': mode, 'round': args.round, 'phase': args.phase, 'scales': scales_report,
                      'output': output_report, 'reference': independent, 'inputs': inputs, 'samples_us': samples}
            report['independent_reference_coverage'] = {'compute_dtype': 'CPU FP32', 'direct_in_this_phase': direct_reference, 'max_rows_per_expert': max(counts) if args.tail_only else 3, 'oracle_rows_per_expert': [min(count, max(counts) if args.tail_only else 3) for count in counts], 'uncovered_rows_per_expert': [max(0, count - (max(counts) if args.tail_only else 3)) for count in counts], 'full_output_scales_comparison': 'all valid elements bitwise against first fresh icpx SOURCE cache; not whole-production independent oracle'}
            results.append(report)
            print(json.dumps(report), flush=True)
            del activations, activation_scales, output, function
        del weights, raw_scales, global_scales, multipliers, exponents, workspace, problems, offsets
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    mappings = [line for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'nvfp4_woq' in line and '.so' in line]
    assert all(str(expected_package) in line for line in mappings)
    assert not any('_port_payload' in line for line in mappings)
    (result_dir / f'round{args.round}-phase{args.phase}-{args.label}.json').write_text(json.dumps({'results': results, 'native_mappings': mappings, 'provider': provider, 'GPU': 9, 'source_only': False, 'normal_AFTER': True, 'canonical_version': 'canonical-v5-icpx', 'accuracy_only': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()

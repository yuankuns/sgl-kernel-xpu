import argparse
import json
from pathlib import Path
import runpy
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.add_argument('--kind', choices=['historical', 'fresh'], required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--evidence', required=True)
    options = parser.parse_args()
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(), 'Only genuine Xe3.5 SOURCE accuracy'
    torch.set_num_threads(16)
    evidence = Path(options.evidence)
    reference = runpy.run_path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/benchmark/accuracy_reference.py')['verify_grouped_output']
    print(json.dumps({'kind': options.kind, 'round': options.round, 'package': sgl_kernel.__file__, 'device': torch.xpu.get_device_name(), 'precision_only': True}), flush=True)
    reports = []
    for allocated_rows, width in [(256, 8192), (256, 4096), (257, 4096), (262144, 4096), (262144, 8192)]:
        torch.xpu.synchronize()
        time.sleep(2)
        torch.manual_seed(42)
        counts = [0] * 256
        if allocated_rows <= 257:
            for expert in torch.randperm(256)[:8 * 8].tolist():
                counts[expert] = 4
        else:
            counts = [allocated_rows // 256] * 256
        rows = torch.tensor(counts, dtype=torch.int32, device='xpu')
        activations = torch.randint(0, 256, (allocated_rows, 2048), dtype=torch.uint8, device='xpu')
        weights = torch.randint(0, 256, (256, width, 2048), dtype=torch.uint8, device='xpu')
        activation_scales = torch.randint(124, 130, (allocated_rows, 128), dtype=torch.uint8, device='xpu')
        weight_scales = torch.randint(124, 130, (256, 128, width), dtype=torch.uint8, device='xpu')
        output = torch.empty((allocated_rows, width), dtype=torch.bfloat16, device='xpu')
        fingerprints = {name: tensor_sha256(tensor) for name, tensor in [('activations', activations), ('weights', weights),
                        ('activation_scales', activation_scales), ('weight_scales', weight_scales), ('rows_per_expert', rows)]}
        sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a4(output, activations, activation_scales, weights, weight_scales, rows, width, 4096, 256)
        torch.xpu.synchronize()
        accuracy = reference(output, activations, activation_scales, weights, weight_scales, counts, max_rows=4)
        valid = output[:sum(counts)]
        output_hash = tensor_sha256(valid)
        shape = f'{allocated_rows}-{width}-4096'
        saved = evidence / f'round-{options.round}-{options.kind}-{shape}.pt'
        torch.save(valid.cpu(), saved)
        reports.append({'shape': shape, 'kind': options.kind, 'round': options.round, 'inputs': fingerprints,
                        'full_valid_output': output_hash, 'output_file': str(saved), 'independent_FP64': accuracy,
                        'oracle_scope': 'max_rows=4 per expert, full columns; not every production row',
                        'unwritten_slack_rows_excluded': allocated_rows - sum(counts), 'performance': 'not run'})
        print(json.dumps(reports[-1]), flush=True)
        del output, valid, activations, weights, activation_scales, weight_scales, rows
    (evidence / f'round-{options.round}-{options.kind}.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()

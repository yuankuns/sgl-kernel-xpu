import argparse
import hashlib
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
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--evidence', required=True)
    options = parser.parse_args()
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    evidence = Path(options.evidence)
    artifact = evidence.parent
    checked = json.loads((evidence / 'normal-package-check.json').read_text())
    assert Path(sgl_kernel.__file__).resolve().parent == Path(checked['package'])
    assert checked['manifest_sha256'] == '6bb13f86e5736f1ebd3d50b59ee67c032ef44511abef71608e212c991de880ba'
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if '/' in line.split()[-1] and ('libsgl-ops-' in line or '/common_ops.' in line or '/libmuse-' in line)}
    assert mapped == {Path(path) for path in checked['mapped_native_hashes']}
    for path in mapped:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == checked['mapped_native_hashes'][str(path)]
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(), 'Only genuine Xe3.5 normal AFTER accuracy'
    torch.set_num_threads(16)
    baseline = json.loads((artifact / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    reference = runpy.run_path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/benchmark/accuracy_reference.py')['verify_grouped_output']
    reports = []
    failed = False
    for index, (allocated_rows, width) in enumerate([(256, 8192), (256, 4096), (257, 4096), (262144, 4096), (262144, 8192)]):
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
        shape = f'{allocated_rows}-{width}-4096'
        assert baseline[index]['shape'] == shape and fingerprints == baseline[index]['inputs']
        sgl_kernel.moe_grouped_mm_nt_mxfp4_w4a4_xe35(output, activations, activation_scales, weights, weight_scales, rows, width, 4096, 256)
        torch.xpu.synchronize()
        valid = output[:sum(counts)]
        output_hash = tensor_sha256(valid)
        saved = evidence / f'round-{options.round}-after-{shape}.pt'
        torch.save(valid.cpu(), saved)
        try:
            accuracy = reference(output, activations, activation_scales, weights, weight_scales, counts, max_rows=4)
            independent_pass = True
        except AssertionError as error:
            accuracy = {'assertion_failure': str(error)}
            independent_pass = False
            failed = True
        reports.append({'shape': shape, 'kind': 'normal-official-AFTER', 'round': options.round, 'inputs': fingerprints,
                        'full_valid_output': output_hash, 'output_file': str(saved), 'independent_FP64': accuracy,
                        'independent_reference_pass': independent_pass,
                        'oracle_scope': 'max_rows=4 per expert, full columns; not every production row',
                        'unwritten_slack_rows_excluded': allocated_rows - sum(counts), 'performance': 'not run',
                        'package': sgl_kernel.__file__, 'actual_registered_API': 'moe_grouped_mm_nt_mxfp4_w4a4_xe35',
                        'normal_manifest_sha256': checked['manifest_sha256']})
        print(json.dumps(reports[-1]), flush=True)
        del output, valid, activations, weights, activation_scales, weight_scales, rows
    (evidence / f'round-{options.round}-after.json').write_text(json.dumps(reports, indent=2) + '\n')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

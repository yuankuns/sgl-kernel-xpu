import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.add_argument('--kind', choices=['source', 'after'], required=True)
    parser.add_argument('--phase', type=int, choices=range(12), required=True)
    parser.add_argument('--warmup', type=int, choices=[5], default=5)
    parser.add_argument('--calls', type=int, choices=[8], default=8)
    parser.add_argument('--repeats', type=int, choices=[31], default=31)
    options = parser.parse_args()
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    directory = Path(__file__).resolve().parent
    assert (directory / 'main-performance-gpu6-grant-v1.json').exists()
    grant = json.loads((directory / 'main-performance-gpu6-grant-v1.json').read_text())
    assert grant['issuer'] == 'main' and grant['release'] and grant['performance_authorized']
    assert grant['external_compilers_drained'] and grant['running_stack_verified']
    assert time.time() >= grant['start_epoch']
    assert sorted(os.sched_getaffinity(0)) == list(range(32, 48)) and os.environ['ZE_AFFINITY_MASK'] == '6'
    assert os.environ['OMP_NUM_THREADS'] == '8'
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch()
    torch.set_num_threads(8)
    check = json.loads((directory / ('CPU-normal-after.json' if options.kind == 'after' else 'CPU-normal-source.json')).read_text())
    expected_package = Path(check['package'])
    assert Path(sgl_kernel.__file__).resolve() == expected_package or Path(sgl_kernel.__file__).resolve().parent == expected_package
    expected_maps = check.get('mapped_native_sha256', check.get('mapped_native_hashes'))
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line}
    assert mapped == {Path(name) for name in expected_maps}
    assert all(digest(path) == expected_maps[str(path)] for path in mapped)
    baseline = json.loads((directory.parent / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    operation = sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a4 if options.kind == 'source' else sgl_kernel.moe_grouped_mm_nt_mxfp4_w4a4_xe35
    reports = []
    for index, (allocated_rows, width) in enumerate([(256, 8192), (256, 4096), (257, 4096), (262144, 4096), (262144, 8192)]):
        if (directory / 'performance-maintenance-hold.json').exists():
            raise SystemExit(125)
        torch.xpu.synchronize()
        time.sleep(2)
        shape = f'{allocated_rows}-{width}-4096'
        assert baseline[index]['shape'] == shape
        shared = directory / f'shared-inputs-scales-{shape}.pt'
        if options.phase == 0:
            assert options.kind == 'source' and not shared.exists()
            torch.manual_seed(42)
            counts = [0] * 256
            if allocated_rows <= 257:
                for expert in torch.randperm(256)[:8 * 8].tolist():
                    counts[expert] = 4
            else:
                counts = [allocated_rows // 256] * 256
            tensors = {'rows_per_expert': torch.tensor(counts, dtype=torch.int32, device='xpu'),
                       'activations': torch.randint(0, 256, (allocated_rows, 2048), dtype=torch.uint8, device='xpu'),
                       'weights': torch.randint(0, 256, (256, width, 2048), dtype=torch.uint8, device='xpu'),
                       'activation_scales': torch.randint(124, 130, (allocated_rows, 128), dtype=torch.uint8, device='xpu'),
                       'weight_scales': torch.randint(124, 130, (256, 128, width), dtype=torch.uint8, device='xpu')}
            fingerprints = {name: tensor_sha256(value) for name, value in tensors.items()}
            assert fingerprints == baseline[index]['inputs']
            torch.save({name: value.cpu() for name, value in tensors.items()}, shared)
            (directory / f'shared-inputs-scales-{shape}.json').write_text(json.dumps({'file_SHA256': digest(shared), 'tensors': fingerprints}) + '\n')
        else:
            record = json.loads((directory / f'shared-inputs-scales-{shape}.json').read_text())
            assert digest(shared) == record['file_SHA256']
            loaded = torch.load(shared, map_location='cpu', weights_only=True, mmap=True)
            assert {name: tensor_sha256(value) for name, value in loaded.items()} == record['tensors'] == baseline[index]['inputs']
            tensors = {name: value.to('xpu') for name, value in loaded.items()}
            del loaded
        valid_rows = int(tensors['rows_per_expert'].sum())
        output = torch.empty((allocated_rows, width), dtype=torch.bfloat16, device='xpu')

        def launch():
            operation(output, tensors['activations'], tensors['activation_scales'], tensors['weights'], tensors['weight_scales'], tensors['rows_per_expert'], width, 4096, 256)

        launch()
        torch.xpu.synchronize()
        assert tensor_sha256(output[:valid_rows]) == baseline[index]['full_valid_output']
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            for call in range(8):
                launch()
        output.fill_(float('nan'))
        torch.xpu.synchronize()
        graph.replay()
        torch.xpu.synchronize()
        replay_hash = tensor_sha256(output[:valid_rows])
        assert replay_hash == baseline[index]['full_valid_output'], 'Graph replay must replace NaN sentinel with every valid original output bit; empty capture rejected'
        for warmup in range(5):
            graph.replay()
        torch.xpu.synchronize()
        samples = []
        graph_samples = []
        for repeat in range(31):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            graph_samples.append(elapsed)
            samples.append(elapsed / 8)
        final_hash = tensor_sha256(output[:valid_rows])
        assert final_hash == baseline[index]['full_valid_output']
        median = statistics.median(samples)
        report = {'shape': shape, 'kind': options.kind, 'phase': options.phase, 'package': sgl_kernel.__file__,
                  'raw_graph_ms': graph_samples, 'raw_per_call_ms': samples, 'median_ms': median,
                  'MAD_ms': statistics.median(abs(value - median) for value in samples),
                  'warmup_graph_replays': 5, 'calls_per_graph': 8, 'measured_repeats': 31,
                  'validation_native_calls': 1, 'validation_graph_replays': 1,
                  'genuine_replay_NaN_sentinel_full_valid_output_proof': True,
                  'input_hashes': baseline[index]['inputs'], 'full_valid_output_hash': final_hash,
                  'valid_rows': valid_rows, 'unwritten_slack_rows_excluded': allocated_rows - valid_rows,
                  'independent_FP64_prevalidated_scope': 'max_rows4/expert/fullcolumns; not every production row',
                  'timing_scope': 'device events around genuine graph replay of eight native calls, elapsed/8; not E2E'}
        reports.append(report)
        (directory / f'phase-{options.phase}-{options.kind}.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        del graph, output, tensors
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


if __name__ == '__main__':
    main()

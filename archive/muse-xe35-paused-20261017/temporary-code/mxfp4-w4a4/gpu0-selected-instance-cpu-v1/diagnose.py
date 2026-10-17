import argparse
import json
import os
from pathlib import Path
import time

from protocol import ARTIFACT, DIRECTORY, PERFORMANCE, SHAPES, digest, dot_launches, normal, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.add_argument('--kind', choices=['source', 'after'], required=True)
    options = parser.parse_args()
    grant = json.loads((DIRECTORY / 'main-gpu0-diagnostic-grant.json').read_text())
    assert grant['issuer'] == 'main' and grant['release'] and grant['gpu'] == 0
    assert grant['diagnostic_only'] and not grant['performance_authorized']
    assert time.time() >= grant['start_epoch'] >= grant['grant_epoch'] + 120
    assert sorted(os.sched_getaffinity(0)) == grant['CPU_affinity']
    assert os.environ['OMP_NUM_THREADS'] == '8' and os.environ['ZE_AFFINITY_MASK'] == '0'
    for name, expected in grant['bound_files_SHA256'].items():
        assert digest(name.replace('/home/syk/', '/root/', 1)) == expected, name
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    assert not torch.xpu.is_initialized()
    provider = normal(options.kind)
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch()
    torch.set_num_threads(8)
    save(DIRECTORY / f'{options.kind}-runtime.json', {'epoch': time.time(), 'PID': os.getpid(), 'provider': provider,
         'CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU': 0, 'diagnostic_not_performance': True})
    baseline = json.loads((ARTIFACT / 'historical-fresh-source-precision-v3/round-1-fresh.json').read_text())
    operation = sgl_kernel.moe_grouped_mm_nt_xe35_mxfp4_w4a4 if options.kind == 'source' else sgl_kernel.moe_grouped_mm_nt_mxfp4_w4a4_xe35
    output = None
    context = {}
    try:
        for shape in SHAPES:
            torch.xpu.synchronize()
            time.sleep(2)
            allocated_rows, width, depth = map(int, shape.split('-'))
            reference = next(row for row in baseline if row['shape'] == shape)
            path = PERFORMANCE / f'shared-inputs-scales-{shape}.pt'
            recorded = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == recorded['file_SHA256']
            loaded = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert {name: tensor_sha256(value) for name, value in loaded.items()} == recorded['tensors'] == reference['inputs']
            tensors = {name: value.to('xpu') for name, value in loaded.items()}
            del loaded
            assert {name: tensor_sha256(value) for name, value in tensors.items()} == reference['inputs']
            valid_rows = int(tensors['rows_per_expert'].sum())
            assert valid_rows == allocated_rows == 262144
            output = torch.empty((allocated_rows, width), dtype=torch.bfloat16, device='xpu')
            pointers = {name: value.data_ptr() for name, value in tensors.items()} | {'output': output.data_ptr()}
            context = {'kind': options.kind, 'shape': shape, 'input_file': str(path), 'input_file_SHA256': recorded['file_SHA256'],
                       'actual_full_input_SHA': reference['inputs'], 'stable_actual_tensor_pointers': pointers}
            save(DIRECTORY / f'{options.kind}-{shape}-before.json', context)

            def launch():
                operation(output, tensors['activations'], tensors['activation_scales'], tensors['weights'], tensors['weight_scales'], tensors['rows_per_expert'], width, depth, 256)

            print(f'W4_DIAG_BEGIN {options.kind} {shape} DIRECT', flush=True)
            launch()
            torch.xpu.synchronize()
            direct_hash = tensor_sha256(output[:valid_rows])
            assert direct_hash == reference['full_valid_output']
            print(f'W4_DIAG_END {options.kind} {shape} DIRECT', flush=True)
            captures = []
            for calls in (1, 8):
                graph = torch.xpu.XPUGraph()
                graph.enable_debug_mode()
                print(f'W4_DIAG_BEGIN {options.kind} {shape} CAPTURE_{calls}', flush=True)
                with torch.xpu.graph(graph):
                    for call_index in range(calls):
                        assert torch.xpu.is_current_stream_capturing()
                        launch()
                dot_path = DIRECTORY / f'{options.kind}-{shape}-capture-{calls}.dot'
                graph.debug_dump(str(dot_path))
                nodes = dot_launches(dot_path.read_text(), options.kind, calls)
                output.fill_(float('nan'))
                torch.xpu.synchronize()
                assert torch.isnan(output[:valid_rows]).all()
                graph.replay()
                torch.xpu.synchronize()
                assert tensor_sha256(output[:valid_rows]) == reference['full_valid_output'], 'real replay must replace every valid NaN with original SOURCE bits'
                assert ({name: value.data_ptr() for name, value in tensors.items()} | {'output': output.data_ptr()}) == pointers
                captures.append({'calls': calls, 'default_keep_graph': False, 'dot_file': str(dot_path), 'DOT_SHA256': digest(dot_path),
                                 'actual_GEMM_nodes': nodes, 'full_valid_output_SHA': reference['full_valid_output'], 'all_valid_NaN_before_replay': True})
                print(f'W4_DIAG_END {options.kind} {shape} CAPTURE_{calls}', flush=True)
                del graph
            save(DIRECTORY / f'{options.kind}-{shape}-result.json', context | {'direct_full_valid_output_SHA': direct_hash, 'captures': captures,
                 'no_warmup_events_or_performance_measurement': True,
                 'FP64_scope': 'prior accepted independent reference <=4rows/expert/fullcolumns; not rerun here or claimed all-production-row FP64',
                 'direct_grid_limitation': 'UR printed dimensions expose first element only; captured DOT provides complete driver groupCounts/groupSizes, no inferred missing direct dimensions'})
            del tensors, output
            output = None
            torch.xpu.empty_cache()
    except BaseException as error:
        if output is not None:
            torch.save({'actual_full_output': output.detach().cpu()}, DIRECTORY / f'{options.kind}-first-failure-output.pt')
        save(DIRECTORY / f'{options.kind}-first-failure.json', context | {'actual_error': repr(error), 'full_inputs_reused_readonly_not_copied': True})
        raise


if __name__ == '__main__':
    main()

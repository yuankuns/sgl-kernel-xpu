import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time

from protocol import ARTIFACT, DIRECTORY, PERFORMANCE, SHAPES, digest, dot_launches, normal, save, ur_direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['source', 'after'], required=True)
    options = parser.parse_args()
    import torch
    from input_forensics import tensor_sha256

    torch.set_num_threads(1)
    assert not torch.xpu.is_initialized()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    provider = normal(options.kind)
    metadata_path = ARTIFACT / 'aot-regression-main-cpu-v1/METADATA_COMPARISON_v3.json'
    metadata = json.loads(metadata_path.read_text())
    assert len(metadata['rows']) == 7
    known = metadata['rows'][0][options.kind]['name']
    table = '<TABLE><TR><TD>kernelName</TD><TD>' + known + '</TD></TR><TR><TD>hKernel</TD><TD>0x1</TD></TR><TR><TD>groupCounts</TD><TD>64x2x256</TD></TR><TR><TD>groupSizes</TD><TD>128x1x1</TD></TR></TABLE>'
    assert len(dot_launches(table, options.kind, 1)) == 1
    assert len(dot_launches(table * 8, options.kind, 8)) == 8
    negatives = {}
    for name, sample, calls in [('missing_grid', table.replace('64x2x256', ''), 1),
                                ('wrong_count', table, 8), ('zero_group', table.replace('64x2x256', '0x2x256'), 1),
                                ('unknown_instance', table.replace(known, 'GemmUniversal_UNKNOWN'), 1)]:
        try:
            dot_launches(sample, options.kind, calls)
        except AssertionError as error:
            negatives[name] = str(error)
        else:
            raise AssertionError('invalid graph evidence accepted: ' + name)
    fixture = ('<--- urKernelCreate(.hProgram = 0x3, .pKernelName = 0x4 (' + known + '), .phKernel = 0x5 (0x6)) -> UR_RESULT_SUCCESS;\n'
               'W4_DIAG_BEGIN ' + options.kind + ' ' + SHAPES[0] + ' DIRECT\n'
               '<--- urEnqueueKernelLaunchWithArgsExp(.hKernel = 0x6, .pGlobalWorkSize = 0x7 (8192), .pLocalWorkSize = 0x8 (128)) -> UR_RESULT_SUCCESS;\n'
               'W4_DIAG_END ' + options.kind + ' ' + SHAPES[0] + ' DIRECT\n')
    assert ur_direct(fixture, SHAPES[0], options.kind)['name'] == known
    try:
        ur_direct(fixture.replace('.phKernel = 0x5 (0x6)', '.phKernel = 0x5 (0x9)'), SHAPES[0], options.kind)
    except AssertionError:
        negatives['unresolved_actual_UR_kernel'] = 'rejected'
    else:
        raise AssertionError('unresolved kernel accepted')
    bound = {}
    for path in DIRECTORY.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        bound[str(path)] = digest(path)
    for path in [metadata_path, ARTIFACT / 'historical-fresh-source-precision-v3/round-1-fresh.json',
                 PERFORMANCE / f'CPU-normal-{options.kind}.json',
                 ARTIFACT / 'official-performance-v4-gpu6-v2-post/W4_TERMINAL_PRECISION_PERFORMANCE_DRAIN_v2.json']:
        bound[str(path)] = digest(path)
    for name, expected in provider['mapped_native_sha256'].items():
        bound[name] = expected
    inputs = {}
    for shape in SHAPES:
        path = PERFORMANCE / f'shared-inputs-scales-{shape}.pt'
        reference = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == reference['file_SHA256']
        tensors = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert {name: tensor_sha256(value) for name, value in tensors.items()} == reference['tensors']
        del tensors
        inputs[shape] = reference
        bound[str(path)] = reference['file_SHA256']
        bound[str(path.with_suffix('.json'))] = digest(path.with_suffix('.json'))
    sycl = Path('/opt/gfx-deps/dpcpp/lib/libsycl.so').resolve()
    loader = Path('/opt/gfx-deps/dpcpp/lib/libur_loader.so').resolve()
    torch_graphs = Path(torch.__file__).parent / 'xpu/graphs.py'
    for path in [sycl, loader, torch_graphs, Path('/opt/gfx-deps/dpcpp/include/unified-runtime/ur_print.hpp')]:
        bound[str(path)] = digest(path)
    for path in [Path('/sys/module/xe/version'), Path('/sys/module/xe/srcversion')]:
        if path.is_file():
            bound[str(path)] = digest(path)
    assert not torch.xpu.is_initialized()
    report = {'actual_CPU_preparation_exit': 0, 'kind': options.kind, 'normal_provider': provider,
              'raw_shared_inputs_hashes': inputs, 'bound_files_SHA256': {name.replace('/root/', '/home/syk/', 1): value for name, value in bound.items()},
              'actual_CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU_initialized': False,
              'parser_CPU_positive_1_and_8_and_negative_cases': negatives, 'parser_fixtures_synthetic_not_GPU_evidence': True,
              'graph_debug_dump_installed_and_other_module_supported_NOT_W4_runtime_proof': hasattr(torch.xpu.XPUGraph, 'enable_debug_mode') and hasattr(torch.xpu.XPUGraph, 'debug_dump'),
              'no_native_compiler_deps_helper_or_PR_mutation': True, 'new_disk_input_copies': 0}
    save(DIRECTORY / f'CPU-{options.kind}.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('normal_provider', 'raw_shared_inputs_hashes', 'bound_files_SHA256')}, indent=2))


if __name__ == '__main__':
    main()

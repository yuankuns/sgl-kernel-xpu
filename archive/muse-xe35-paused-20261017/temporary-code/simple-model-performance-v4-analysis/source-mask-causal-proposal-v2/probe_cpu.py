import hashlib
import inspect
import json
import os
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    import torch
    assert not torch.xpu.is_initialized()
    graph_source = Path(inspect.getsourcefile(torch.xpu.XPUGraph))
    names = ('capture_begin', 'capture_end', 'instantiate', 'replay', 'reset', 'debug_dump', 'raw_xpu_graph', 'raw_xpu_graph_exec')
    methods = {name: {'exists': hasattr(torch.xpu.XPUGraph, name), 'callable': callable(getattr(torch.xpu.XPUGraph, name, None)), 'doc': inspect.getdoc(getattr(torch.xpu.XPUGraph, name, None))} for name in names}
    tools = {}
    for root in (Path('/opt/gfx-deps/profiling_tools'), Path('/opt/gfx-deps/dpcpp')):
        for relative in ('bin/unitrace', 'bin/onetrace', 'bin/urtrace', 'bin/ze_tracer', 'lib/libze_loader.so', 'lib/libsycl.so'):
            path = root / relative
            if path.is_file():
                tools[str(path)] = {'sha256': digest(path), 'tool_not_executed': True}
    torch_header = Path(torch.__file__).parent / 'include/ATen/xpu/XPUGraph.h'
    text = torch_header.read_text() if torch_header.is_file() else ''
    header_lines = [line.strip() for line in text.splitlines() if any(token in line for token in ('instantiate', 'raw_xpu', 'raw_graph', 'keep_graph', 'replay', 'debug_dump', 'command_graph', 'graph_'))]
    sycl_roots = (Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/experimental/graph'), Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/experimental'))
    graph_headers = {}
    for root in sycl_roots:
        for name in ('command_graph.hpp', 'node.hpp', 'dynamic_parameter.hpp', 'graph.hpp'):
            path = root / name
            if path.is_file():
                graph_headers[str(path)] = {'sha256': digest(path), 'get_native_argument_value_API_claimed': False}
    assert not torch.xpu.is_initialized()
    result = {'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'Torch': torch.__version__, 'Torch_provider': torch.__file__, 'Torch_XPU_initialized': False, 'GPU_started': False, 'compiler_started': False, 'Torch_graph_Python_source': str(graph_source), 'Torch_graph_Python_source_sha256': digest(graph_source), 'methods': methods, 'installed_XPUGraph_header': str(torch_header), 'installed_XPUGraph_header_SHA256': digest(torch_header) if torch_header.is_file() else None, 'installed_XPUGraph_header_relevant_declarations': header_lines, 'tools_static_existence_only': tools, 'SYCL_graph_headers_static_only': graph_headers, 'raw_graph_handle_not_dereferenced': True, 'runtime_trace_available_or_working_claimed': False, 'native_argument_scalar_values_observed': False}
    with (DIRECTORY / 'INSTALLED_API_CPU_PROBE.json').open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'Torch_XPU_initialized': False, 'raw_xpu_graph_API_exists': methods['raw_xpu_graph']['exists'], 'static_tools': list(tools), 'runtime_trace_claimed': False}))


if __name__ == '__main__':
    main()

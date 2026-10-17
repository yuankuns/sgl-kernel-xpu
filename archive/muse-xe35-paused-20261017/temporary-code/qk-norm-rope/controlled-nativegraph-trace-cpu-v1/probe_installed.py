import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'controlled-method-runner-cpu-v1'))
from protocol import digest, require, save


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'CPU0–15 only')
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU investigation cannot initialize XPU')
    activities = sorted(str(value) for value in torch.profiler.supported_activities())
    require(not torch.xpu.is_initialized(), 'Capability lookup unexpectedly initialized XPU')
    package = Path(torch.__file__).resolve().parent
    files = [package / 'xpu/graphs.py', package / 'profiler/profiler.py', package / 'autograd/profiler.py',
        package / 'include/ATen/xpu/XPUGraph.h', package / 'include/torch/csrc/profiler/stubs/xpu.h',
        package / 'include/torch/csrc/profiler/kineto_shim.h']
    inventory = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size} for path in files if path.is_file()}
    native = package / 'lib/libtorch_xpu.so'
    command = ['readelf', '-d', str(native)]
    result = subprocess.run(command, capture_output=True, text=True)
    require(result.returncode == 0, 'Read-only installed native dependencies inspection failed')
    (ROOT / 'libtorch-xpu-needed.log').write_text(result.stdout)
    pti = Path('/opt/gfx-deps/profiling_tools')
    headers = [path for path in pti.glob('include/**/*.h') if any(token in path.name.lower() for token in ('view', 'version'))]
    libraries = sorted({path.resolve() for path in (pti / 'lib').glob('*pti*.so*')})
    environment = {name: os.environ.get(name) for name in ('ZE_ENABLE_TRACING_LAYER', 'UR_ENABLE_LAYERS', 'UR_LOG_LOADER',
        'SYCL_PI_TRACE', 'SYCL_UR_TRACE', 'KINETO_LOG_LEVEL', 'PTI_LOG_LEVEL', 'PTI_ENABLE_COLLECTION',
        'ONEAPI_DEVICE_SELECTOR', 'SYCL_DEVICE_FILTER', 'ZE_AFFINITY_MASK', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    value = {'actual_CPU_probe_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'XPU_initialized': torch.xpu.is_initialized(),
        'Torch_version': torch.__version__, 'Torch_git_version': torch.version.git_version,
        'Torch_config': torch.__config__.show(), 'kineto_available': torch.autograd.kineto_available(),
        'supported_activities': activities, 'XPUGraph_methods_present': {name: hasattr(torch.xpu.XPUGraph, name) for name in
            ('capture_begin', 'instantiate', 'replay', 'debug_dump', 'raw_xpu_graph', 'raw_xpu_graph_exec')},
        'ExperimentalConfig_documentation': torch._C._profiler._ExperimentalConfig.__doc__,
        'installed_source_headers_SHA': inventory, 'native_readelf_command': command,
        'profiling_tool_headers_SHA': {str(path): digest(path) for path in headers},
        'profiling_tool_library_files': [{'path': str(path), 'bytes': path.stat().st_size} for path in libraries],
        'actual_CPU_probe_instrumentation_environment_NOT_failed_phase_environment': environment,
        'no_profiler_session_graph_capture_replay_device_query_or_GPU_kernel_executed': True}
    save(ROOT / 'installed-capabilities.json', value)
    print(json.dumps({'actual_exit': 0, 'supported_activities': activities, 'kineto_available': value['kineto_available'],
        'XPU_initialized': value['XPU_initialized'], 'PTI_libraries': len(libraries)}))


if __name__ == '__main__':
    main()

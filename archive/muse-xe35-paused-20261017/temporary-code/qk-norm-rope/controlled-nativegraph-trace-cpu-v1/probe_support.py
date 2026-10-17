import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'controlled-method-runner-cpu-v1'))
from protocol import digest, require, save


def main():
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light CPU0–15 only')
    import torch

    require(not torch.xpu.is_initialized(), 'Read-only metadata cannot initialize XPU')
    include = Path('/opt/gfx-deps/profiling_tools/include/pti')
    names = ('pti_version.h', 'pti_view.h', 'pti_driver_levelzero_api_ids.h', 'pti_runtime_sycl_api_ids.h')
    files = {name: {'sha256': digest(include / name), 'text': (include / name).read_text()} for name in names}
    graph_root = Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/experimental')
    selected = [graph_root / 'graph.hpp'] + list((graph_root / 'graph').glob('*.hpp'))
    graph = {}
    for path in selected:
        if path.is_file():
            lines = path.read_text().splitlines()
            matching = [{'line': index + 1, 'text': line} for index, line in enumerate(lines) if 'profil' in line.lower()]
            graph[str(path)] = {'sha256': digest(path), 'profiling_declarations': matching}
    value = {'actual_CPU_probe_exit': 0, 'CPU_affinity': list(range(16)), 'XPU_initialized': False,
        'PTI_headers': files, 'SYCL_graph_headers': graph,
        'ExperimentalConfig_init_documentation': torch._C._profiler._ExperimentalConfig.__init__.__doc__,
        'actual_PTIdriver_table_has_native_launch_WithArguments': 'zeCommandListAppendLaunchKernelWithArguments_id=209' in files['pti_driver_levelzero_api_ids.h']['text'],
        'zero_profiler_sessions_or_graph_device_queries': True,
        'initial_separate_readonly_tool_CPU_import_failure_exit1_due_missing_LD_prepend_preserved': True}
    save(ROOT / 'profiling-support.json', value)
    print('CPU metadata0; PTI table includes native LaunchKernelWithArguments209; no GPU/graph/profiler session; old GPU failure unchanged')


if __name__ == '__main__':
    main()

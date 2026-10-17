import json
import os

from graphs import profiler_state
from protocol import require

require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'CPU0–15 only')
import torch

torch.set_num_threads(1)
require(not torch.xpu.is_initialized(), 'No XPU initialized')
state = profiler_state(torch)
print(json.dumps({'actual_exit': 0, 'torch_file': torch.__file__, 'torch_version': torch.__version__,
    'actual_installed_profiler_state_API': state, 'XPU_initialized': torch.xpu.is_initialized(),
    'profiler_session_started': False, 'GPU_authorized': False}, indent=2))

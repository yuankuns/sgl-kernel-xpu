import json
import os


inherited = os.environ.get('LD_LIBRARY_PATH', '')
assert inherited.split(':')[0] == '/opt/gfx-deps/profiling_tools/lib'

import torch

initialized_before_cpu = torch.xpu.is_initialized()
assert not initialized_before_cpu
result = torch.ones(4, device='cpu').sum().item()
assert not torch.xpu.is_initialized()
print(json.dumps({'Torch_CPU_import': True, 'torch_version': torch.__version__, 'CPU_result': result,
                  'xpu_initialized': False, 'preexported_LD_LIBRARY_PATH': inherited,
                  'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH'], 'GPU_work': False}), flush=True)

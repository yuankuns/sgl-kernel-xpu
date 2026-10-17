import hashlib
import json
import os
from pathlib import Path

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not torch.xpu.is_initialized() and sorted(os.sched_getaffinity(0)) == list(range(64, 80))
paths = [Path('/opt/gfx-deps/dpcpp/bin/icpx'), Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.39229'),
         Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0'), Path('/opt/gfx-deps/dpcpp/lib/libsycl.so.9.0.0-0'),
         Path('/opt/gfx-deps/dpcpp/lib/libur_adapter_level_zero.so.0.12.0'), Path(torch.__file__)]
report = {'GPU_initialized': False, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
          'Torch_provider': torch.__file__, 'driver_runtime_compiler_SHA256': {str(path): digest(path) for path in paths}}
directory = Path(__file__).resolve().parent
(directory / 'CPU-driver-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

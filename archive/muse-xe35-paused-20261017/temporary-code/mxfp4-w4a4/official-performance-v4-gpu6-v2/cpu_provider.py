import hashlib
import json
import os
from pathlib import Path

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
assert sorted(os.sched_getaffinity(0)) == list(range(32, 48)) and not torch.xpu.is_initialized()
old = json.loads((directory.parent / 'official-after-precision-v4-gpu11-v2/CPU-provider.json').read_text())
assert digest(Path(torch.__file__)) == old['provider_init_sha256']
assert digest(Path('/opt/gfx-deps/dpcpp/bin/icpx')) == old['icpx_binary_sha256']
for name, expected in old['driver_runtime_and_native_mapped_sha256'].items():
    assert digest(Path(name)) == expected, name
report = {'GPU_initialized': False, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
          'Torch_provider': torch.__file__, 'Torch_version': torch.__version__,
          'unchanged_driver_compiler_native_SHA256': old['driver_runtime_and_native_mapped_sha256'],
          'icpx_binary_SHA256': old['icpx_binary_sha256'], 'provider_init_SHA256': old['provider_init_sha256']}
(directory / 'CPU-driver-provider.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

import hashlib
import json
import os
from pathlib import Path

import torch
import sgl_kernel


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
checked = json.loads((directory / 'normal-package-check.json').read_text())
assert not torch.xpu.is_initialized()
assert list(sorted(os.sched_getaffinity(0))) == list(range(64, 80))
assert Path(sgl_kernel.__file__).resolve().parent == Path(checked['package'])
assert Path(torch.__file__).resolve().is_relative_to('/opt/gfx-deps/venv')
paths = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
         if '/' in line.split()[-1] and any(name in line for name in ['libze', 'libur_', 'libsycl', 'libsgl-ops-', '/common_ops.'])}
report = {'CPU_affinity': list(sorted(os.sched_getaffinity(0))), 'GPU_initialized': torch.xpu.is_initialized(),
          'torch_provider': torch.__file__, 'normal_package': sgl_kernel.__file__,
          'provider_init_sha256': digest(Path(torch.__file__)),
          'icpx_binary_sha256': digest(Path('/opt/gfx-deps/dpcpp/bin/icpx')),
          'driver_runtime_and_native_mapped_sha256': {str(path): digest(path) for path in sorted(paths)}}
(directory / 'CPU-provider.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

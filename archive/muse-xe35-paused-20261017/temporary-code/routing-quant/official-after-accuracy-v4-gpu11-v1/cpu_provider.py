import hashlib
import json
import os
from pathlib import Path

import torch
import sgl_kernel


directory = Path(__file__).resolve().parent
check = json.loads((directory / 'normal-after-precheck.json').read_text())
assert sorted(os.sched_getaffinity(0)) == list(range(72, 80))
assert not torch.xpu.is_initialized()
assert Path(sgl_kernel.__file__).resolve().parent == Path(check['package'])
previous = json.loads((directory.parent / 'fresh-three-source-diagnostic-v1/CPU-driver-provenance.json').read_text())
for name, expected in previous['driver_runtime_compiler_SHA256'].items():
    with Path(name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
report = {'package': sgl_kernel.__file__, 'mapped_native_sha256': check['mapped_native_hashes'],
          'Torch_provider': torch.__file__, 'Torch_version': torch.__version__, 'GPU_initialized': False,
          'CPU_affinity': sorted(os.sched_getaffinity(0)), 'driver_runtime_compiler_SHA256': previous['driver_runtime_compiler_SHA256'],
          'normal_full_current_manifest_SHA256': check['manifest_sha256'], 'normal_full_source_digest': check['source_digest']}
(directory / 'fixture-provider-precheck.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

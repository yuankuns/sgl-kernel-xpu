import hashlib
import json
import os
from pathlib import Path

import sgl_kernel
import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
stage = directory.parent / 'fresh-current-source-icpx-native-v3'
assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
assert not torch.xpu.is_initialized()
assert Path(sgl_kernel.__file__).resolve() == stage / 'site/sgl_kernel/__init__.py'
driver = json.loads((directory.parent / 'current-source-accuracy-gpu8-v1/CPU-driver-provenance.json').read_text())
for name, expected in driver['driver_runtime_compiler_SHA256'].items():
    assert digest(Path(name)) == expected, name
accepted = json.loads((stage / 'cpu-native-acceptance.json').read_text())
for name, expected in accepted['actual_current_headers_SHA256'].items():
    assert digest(Path(name)) == expected, name
maps = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line}
assert maps == accepted['current_mapped_native_SHA256'].keys()
for name, expected in accepted['current_mapped_native_SHA256'].items():
    assert digest(Path(name)) == expected, name
manifest = json.loads((stage / 'source-native-package-manifest.json').read_text())
for name, expected in manifest['current_install_files'].items():
    assert digest(stage / 'site' / name) == expected, name
protected = json.loads((stage / 'source-preverify.json').read_text())
source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
dependency = Path('/root/worktrees/cutlass_internal/muse-w4a4-e87870b')
for root, inventory in [(source, protected['original_tracked']), (dependency, protected['dependency_tracked'])]:
    for name, expected in inventory.items():
        assert digest(root / name) == expected, name
assert not torch.xpu.is_initialized()
print(json.dumps({'actual_CPU_preGPU_verification_exit': 0, 'GPU_initialized': False,
                  'normal_SOURCE_site': str(stage / 'site'), 'mapped_native_count': len(maps),
                  'four_fresh_native_SHA256': accepted['four_tested_native_SHA256']}))

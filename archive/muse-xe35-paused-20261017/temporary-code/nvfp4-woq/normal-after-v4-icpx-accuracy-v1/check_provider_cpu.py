import hashlib
import json
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
sys.path.insert(0, str(ARTIFACT))
import canonical_check


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


record = canonical_check.check('canonical-v4-icpx')
assert record['manifest_sha256'] == 'f4f407d06c17ddfb32d56c8392847bf76920861a8244defef403691e7700543a'
site = ARTIFACT / 'canonical-v4-icpx/site'
sys.path.insert(0, str(site))
import torch

assert not torch.xpu.is_initialized()
expected_torch = Path('/opt/gfx-deps/venv/lib/python3.12/site-packages/torch').resolve()
assert Path(torch.__file__).resolve().parent == expected_torch
assert all('target-venv' not in path for path in sys.path)
import sgl_kernel

assert Path(sgl_kernel.__file__).resolve() == (site / 'sgl_kernel/__init__.py').resolve()
assert Path(sgl_kernel.common_ops.__file__).resolve().parent == (site / 'sgl_kernel').resolve()
manifest_path = ARTIFACT / 'canonical-v4-icpx/source-binary-manifest.json'
manifest = json.loads(manifest_path.read_text())
required = ('libsgl-ops-sycl-nvfp4_woq.so', 'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so', 'libsgl-ops-sycl-nvfp4_woq_scale_layout.so')
mapped = {}
for line in Path('/proc/self/maps').read_text().splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) == 6 and fields[5].startswith('/'):
        path = Path(fields[5]).resolve()
        if path.name in required:
            assert path.parent == (site / 'sgl_kernel').resolve()
            metadata = manifest['binaries'][path.name]
            assert path == (site / metadata['path']).resolve()
            assert digest(path) == metadata['sha256']
            mapped[path.name] = {'path': str(path), 'sha256': metadata['sha256']}
        if path.name.startswith(('libtorch', 'libc10')):
            assert path.parent == expected_torch / 'lib'
assert set(mapped) == set(required)
schemas = {name: str(getattr(torch.ops.sgl_kernel, name).default._schema) for name in ('nvfp4_woq_prep_xe35', 'nvfp4_woq_bf16_grouped_mm_xe35', 'nvfp4_woq_mxfp8_scaled_grouped_mm_xe35', 'mxfp4_moe_scale_to_mn_major_xe35')}
assert not torch.xpu.is_initialized()
result = {'actual_CPU_provider_exit': 0, 'normal_package_import_verified': True, 'current_normal_default_all_verified': True, 'GPU_initialized': False, 'kernel_invoked': False, 'site': str(site), 'torch': str(expected_torch), 'torch_C_sha256': digest(Path(torch._C.__file__)), 'torch_xpu_sha256': digest(expected_torch / 'lib/libtorch_xpu.so'), 'canonical_check': record, 'current_manifest_sha256': digest(manifest_path), 'current_compiler_policy_sha256': digest(ARTIFACT / 'canonical-v4-icpx/compiler-policy-verified.json'), 'native': mapped, 'schemas': schemas, 'driver_sha256': digest(Path(__file__)), 'GPU_accuracy_verified': False, 'performance_verified': False}
print(json.dumps(result, indent=2))

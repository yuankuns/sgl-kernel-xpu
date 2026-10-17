import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path('/root' if str(ROOT).startswith('/root/') else '/home/syk')
SOURCE = HOME / 'worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014'
SITE = ROOT / 'native-diagnostic/source/site'
REVISION = 'd983e877d6bf97015027bd3b1379ff047009a5dd'
EXPECTED = {
    'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so': '864f695ee11647af36ea1140c941d3ce6f68ae3c1c22a5928784d34766ff4e06',
    'libsgl-ops-sycl-nvfp4_woq.so': 'bad2595a97d75d903fecac9770ddc2bf659baaee1cff362c84204108b6c131fa',
    'libsgl-ops-sycl-mxfp4_moe_scale_layout.so': '2196b48e0e601bd78f9610bc2b6f5541c8942ba7b5e78d30af2946acef33e3d4',
}
SIGNATURES = {
    'libsgl-ops-sycl-nvfp4_woq.so': ['at::native::xpu::nvfp4_woq_prep(at::Tensor&, at::Tensor&, at::Tensor const&, at::Tensor const&)'],
    'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so': ['nvfp4_woq_bf16_grouped_mm(at::Tensor&, ' + ', '.join(['at::Tensor const&'] * 7) + ')', 'nvfp4_woq_mxfp8_scaled_grouped_mm(at::Tensor&, ' + ', '.join(['at::Tensor const&'] * 8) + ')'],
    'libsgl-ops-sycl-mxfp4_moe_scale_layout.so': ['mxfp4_moe_scale_to_mn_major(at::Tensor&, at::Tensor const&, at::Tensor const&)'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(runtime=False):
    git = ['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE)]
    assert subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(git + ['diff', REVISION, '--', 'src', 'include', 'python', 'tests', 'benchmark'], text=True)
    package = SITE / 'sgl_kernel'
    native = {}
    for name, expected in EXPECTED.items():
        path = package / name
        assert digest(path) == digest(SOURCE / 'build/src' / name) == expected
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', '-C', str(path)], text=True)
        public = [line.split(' T ', 1)[1] for line in symbols.splitlines() if ' T ' in line]
        assert all(public.count(signature) == 1 for signature in SIGNATURES[name]), name
        dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        assert f'Library soname: [{name}]' in dynamic
        assert '_port_payload' not in dynamic
        native[name] = {'sha256': expected, 'original_abi': SIGNATURES[name], 'dt_needed': re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)}
    registrations = list(package.glob('common_ops*.so'))
    assert len(registrations) == 1
    registration_sha = digest(registrations[0])
    assert registration_sha == 'e63d50179a696799f4e6b57bb60d50956282b37537a788ffa05581dc85cd8fea'
    dynamic = subprocess.check_output(['readelf', '-d', str(registrations[0])], text=True)
    needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)
    assert set(EXPECTED).issubset(needed)
    python_files = {}
    for name in ('nvfp4.py', 'cutlass_moe.py'):
        original = subprocess.check_output(git + ['show', f'{REVISION}:python/sgl_kernel/{name}'])
        assert digest(package / name) == hashlib.sha256(original).hexdigest()
        python_files[name] = digest(package / name)
    report = {'source_revision': REVISION, 'source_site': str(SITE), 'native': native, 'python_d983_files': python_files, 'registration_sha256': registration_sha, 'registration_full_package_revision': '63fae55', 'entire_registration_package_is_d983': False, 'original_native_abi_verified': True, 'runtime_import_verified': False, 'runtime_maps_verified': False, 'gpu_started': False}
    if runtime:
        for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
            if name in os.environ and os.environ[name] != expected:
                raise RuntimeError(f'{name} must be {expected} before GPU initialization')
        assert os.environ.get('ZE_AFFINITY_MASK') in ('9', '5')
        import sgl_kernel
        import torch
        assert pathlib.Path(sgl_kernel.__file__).resolve() == package / '__init__.py'
        assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch()
        schemas = {name: str(getattr(torch.ops.sgl_kernel, name).default._schema) for name in ('nvfp4_woq_prep', 'nvfp4_woq_bf16_grouped_mm', 'nvfp4_woq_mxfp8_scaled_grouped_mm')}
        mapped = {line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'nvfp4_woq' in line and '.so' in line}
        assert mapped and all(str(package) in path for path in mapped)
        for name in EXPECTED:
            assert any(pathlib.Path(path).name == name for path in pathlib.Path('/proc/self/maps').read_text().split() if path.startswith('/'))
        report.update({'runtime_import_verified': True, 'runtime_maps_verified': True, 'schemas': schemas, 'gpu': os.environ['ZE_AFFINITY_MASK'], 'genuine_gpu_query_performed': True})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.runtime), indent=2))

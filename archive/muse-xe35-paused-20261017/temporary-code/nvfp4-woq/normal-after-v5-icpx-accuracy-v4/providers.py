import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
CANONICAL = ARTIFACT / 'canonical-v5-icpx'
FRESH = ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017'
SOURCE_ACCURACY = ARTIFACT / 'source-native-icpx-v2-accuracy-v1'
SOURCE_READY_SHA = 'c4d3fbf54ac0935032593c8ed59d86863da29abfaa3bb1743b92a65b281ce160'
SITES = {'source': FRESH / 'site', 'after': CANONICAL / 'site'}
PYTHONS = {'source': Path('/opt/gfx-deps/venv/bin/python'), 'after': Path('/opt/gfx-deps/venv/bin/python')}
TORCH_ROOT = Path('/opt/gfx-deps/venv/lib/python3.12/site-packages/torch')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preflight():
    for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if name in os.environ and os.environ[name] != expected:
            raise RuntimeError(name + ' rejected before Torch/GPU initialization')


def canonical_result():
    import canonical_check_v5

    return canonical_check_v5.check('canonical-v5-icpx')


def source_result():
    ready_path = SOURCE_ACCURACY / 'SOURCE_READY.json'
    assert digest(ready_path) == SOURCE_READY_SHA
    ready = json.loads(ready_path.read_text())
    assert ready['READY_for_fresh_SOURCE_accuracy_baseline']
    assert ready['host_actual_exit'] == 0 and ready['all_known_owned_descendants_postdrained']
    assert ready['actual_GPU9_wrapper_lock_released'] and ready['GPU9_nonblocking_lock_probe_exit'] == 0
    assert ready['original_precision_tests'] == {'historical': 50, 'fresh_icpx': 50, 'skipped': 0}
    assert (SOURCE_ACCURACY / 'postcheck-CPU.exit').read_text().strip() == '0'
    assert (SOURCE_ACCURACY / 'host-supervisor.exit').read_text().strip() == '0'
    assert (SOURCE_ACCURACY / 'GPU-wrapper.exit').read_text().strip() == '0'
    summary = json.loads((SOURCE_ACCURACY / 'summary.json').read_text())
    assert digest(SOURCE_ACCURACY / 'summary.json') == ready['postcheck']['summary_sha256']
    for family in ('tails', 'production'):
        assert summary['families'][family]['records'] == 16
        assert summary['families'][family]['full_valid_outputs_and_scales_bitwise']
        assert summary['families'][family]['timing_samples'] == 0
    return ready


def runtime_grant():
    preflight()
    path = os.environ.get('NV_V5_ACCURACY_GRANT')
    if not path:
        raise RuntimeError('Main v5 futureepoch accuracy grant required before GPU initialization')
    from supervisor import verify_grant

    record = verify_grant(Path(path))
    assert os.environ.get('ZE_AFFINITY_MASK') == '9' and time.time() >= record['start_epoch']
    started = json.loads((DIRECTORY / 'GPU-suite-started.json').read_text())
    assert started['grant_sha256'] == digest(Path(path)) and started['epoch'] >= record['start_epoch']
    return record


def verify_provider(label, cpu_only=False):
    preflight()
    source_result()
    result = canonical_result()
    import torch
    import sgl_kernel

    expected_torch = TORCH_ROOT.resolve()
    assert Path(torch.__file__).resolve().parent == expected_torch
    assert Path(sys.executable).absolute() == PYTHONS[label].absolute()
    assert all('target-venv' not in path for path in sys.path)
    site = SITES[label].resolve()
    foreign_site = SITES['after' if label == 'source' else 'source'].resolve()
    assert all(Path(path).resolve() != foreign_site for path in sys.path if path)
    assert Path(sgl_kernel.__file__).resolve().parent == site / 'sgl_kernel'
    assert Path(sgl_kernel.common_ops.__file__).resolve().parent == site / 'sgl_kernel'
    if label == 'source':
        manifest = json.loads((FRESH / 'source-binary-manifest.json').read_text())
        assert manifest['source_revision'] == 'd983e877d6bf97015027bd3b1379ff047009a5dd'
        assert manifest['source_cutlass_revision'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23'
        expected = manifest['package_files_sha256']
        native = {Path(name).name: {'path': name, 'sha256': checksum} for name, checksum in expected.items() if name.endswith('.so')}
        assert digest(Path(sgl_kernel.common_ops.__file__)) == manifest['registration_sha256']
    else:
        manifest = json.loads((CANONICAL / 'source-binary-manifest.json').read_text())
        expected = {name: metadata['sha256'] for name, metadata in (manifest['installed_files'] | manifest['python_snapshot_files']).items()}
        native = manifest['binaries']
    actual = {str(path.relative_to(site)): digest(path) for path in site.rglob('*') if path.is_file()}
    assert actual == expected
    mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5]).resolve()
            if path.name.startswith(('libtorch', 'libc10')):
                assert path.parent == expected_torch / 'lib', 'Wrong Torch native provider: ' + str(path)
            if path.name.startswith('libsgl-ops-') or path.name.startswith('common_ops.'):
                assert path.name in native and path == (site / native[path.name]['path']).resolve()
                assert digest(path) == native[path.name]['sha256']
                mapped[path.name] = {'path': str(path), 'sha256': native[path.name]['sha256']}
    required = set(manifest['native']) if label == 'source' else set(manifest['common_ops_dt_needed']) & set(native)
    if label == 'after':
        required |= {Path(manifest['common_ops']).name, 'libsgl-ops-sycl-nvfp4_woq.so', 'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so', 'libsgl-ops-sycl-nvfp4_woq_scale_layout.so'}
    else:
        registration_path = Path(sgl_kernel.common_ops.__file__)
        mapped[registration_path.name] = {'path': str(registration_path), 'sha256': digest(registration_path)}
    assert required.issubset(mapped)
    suffix = '_xe35' if label == 'after' else ''
    names = ('nvfp4_woq_prep', 'nvfp4_woq_bf16_grouped_mm', 'nvfp4_woq_mxfp8_scaled_grouped_mm')
    if label == 'source':
        names += ('mxfp4_moe_scale_to_mn_major',)
    schemas = {name + suffix: str(getattr(torch.ops.sgl_kernel, name + suffix).default._schema) for name in names}
    if label == 'after':
        import subprocess

        helper = site / native['libsgl-ops-sycl-nvfp4_woq_scale_layout.so']['path']
        exports = subprocess.run(['nm', '-D', '--defined-only', '--demangle', str(helper)], check=True, capture_output=True, text=True).stdout
        assert 'mxfp4_moe_scale_to_mn_major_xe35(' in exports, 'Actual scale helper native export missing'
    if cpu_only:
        assert not torch.xpu.is_initialized()
    return {'label': label, 'python': sys.executable, 'torch': str(expected_torch), 'torch_version': torch.__version__, 'torch_C_sha256': digest(Path(torch._C.__file__)), 'torch_xpu_sha256': digest(expected_torch / 'lib/libtorch_xpu.so'), 'package': sgl_kernel.__file__, 'registration': sgl_kernel.common_ops.__file__, 'native_mappings': mapped, 'schemas': schemas, 'GPU_initialized': torch.xpu.is_initialized(), 'canonical_version': 'canonical-v5-icpx', 'after_manifest_sha256': result['manifest_sha256'], 'compiler_policy_sha256': result['compiler_policy_sha256'], 'SOURCE_READY_sha256': SOURCE_READY_SHA, 'source_binary_manifest_sha256': digest(FRESH / 'source-binary-manifest.json'), 'normal_AFTER': label == 'after'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=('source', 'after'))
    parser.add_argument('--version', choices=('canonical-v5-icpx',), default='canonical-v5-icpx')
    args = parser.parse_args()
    print(json.dumps(verify_provider(args.label, cpu_only=True), indent=2))

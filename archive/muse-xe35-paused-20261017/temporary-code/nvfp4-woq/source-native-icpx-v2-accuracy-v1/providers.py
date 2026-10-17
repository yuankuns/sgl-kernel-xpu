import hashlib
import json
import os
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
FRESH = ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017'
SITES = {'historical': ARTIFACT / 'native-diagnostic/source/site', 'fresh': FRESH / 'site'}
PYTHONS = {'historical': ARTIFACT / 'target-venv/bin/python', 'fresh': Path('/opt/gfx-deps/venv/bin/python')}
TORCH_ROOTS = {'historical': ARTIFACT / 'target-venv/lib/python3.12/site-packages/torch', 'fresh': Path('/opt/gfx-deps/venv/lib/python3.12/site-packages/torch')}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preflight():
    for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if name in os.environ and os.environ[name] != expected:
            raise RuntimeError(name + ' rejected before GPU initialization')


def verify_provider(label, cpu_only=False):
    preflight()
    import torch
    import sgl_kernel

    expected_torch = TORCH_ROOTS[label].resolve()
    assert Path(torch.__file__).resolve().parent == expected_torch
    assert Path(sys.executable).absolute() == PYTHONS[label].absolute()
    expected_site = SITES[label].resolve()
    assert Path(sgl_kernel.__file__).resolve().parent == expected_site / 'sgl_kernel'
    foreign = TORCH_ROOTS['fresh' if label == 'historical' else 'historical'].parent.resolve()
    assert all(Path(path).resolve() != foreign for path in sys.path if path)
    manifest = json.loads((DIRECTORY / 'historical-package-manifest.json' if label == 'historical' else FRESH / 'source-binary-manifest.json').read_text())
    actual = {str(path.relative_to(expected_site)): digest(path) for path in expected_site.rglob('*') if path.is_file()}
    assert actual == manifest['package_files_sha256']
    mappings = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            mapped = Path(fields[5]).resolve()
            if mapped.name.startswith(('libtorch', 'libc10')):
                assert mapped.parent == expected_torch / 'lib', 'Foreign Torch native provider: ' + str(mapped)
            if mapped.name in manifest['native']:
                assert mapped == expected_site / 'sgl_kernel' / mapped.name
                assert digest(mapped) == manifest['native'][mapped.name]['sha256']
                mappings.append(str(mapped))
    assert set(Path(path).name for path in mappings) == set(manifest['native'])
    if cpu_only:
        assert not torch.xpu.is_initialized()
    return {'label': label, 'python': sys.executable, 'torch': str(expected_torch), 'torch_version': torch.__version__, 'torch_C_sha256': digest(Path(torch._C.__file__)), 'torch_xpu_sha256': digest(expected_torch / 'lib/libtorch_xpu.so'), 'package': sgl_kernel.__file__, 'registration': sgl_kernel.common_ops.__file__, 'native_mappings': sorted(set(mappings)), 'GPU_initialized': torch.xpu.is_initialized(), 'PORT_or_AFTER': False}


if __name__ == '__main__':
    print(json.dumps(verify_provider(sys.argv[1], cpu_only=True), indent=2))

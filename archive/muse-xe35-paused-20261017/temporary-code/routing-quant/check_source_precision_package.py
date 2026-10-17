import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['historical', 'fresh'], required=True)
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    artifact = Path(__file__).resolve().parent
    stage = artifact / 'fresh-source-icpx-native-v2'
    if options.kind == 'historical':
        site = artifact / 'historical-source-normal-v1/site'
        assert int((artifact / 'historical-source-normal-v1-install.exit').read_text()) == 0
        wheel = next((artifact / 'source-wheels').glob('*.whl'))
        original = json.loads((artifact / 'source-package-provenance.json').read_text())
        assert digest(wheel) == original['wheel_sha256']
        expected = {}
        with zipfile.ZipFile(wheel) as package:
            for name in package.namelist():
                if name.startswith('sgl_kernel/') and not name.endswith('/'):
                    checksum = hashlib.sha256()
                    with package.open(name) as stream:
                        for block in iter(lambda: stream.read(8 * 1024**2), b''):
                            checksum.update(block)
                    expected[name] = checksum.hexdigest()
        provenance = artifact / 'source-before-provenance-v3-icpx.json'
    else:
        assert int((stage / 'cpu-native-acceptance.exit').read_text()) == 0
        manifest_path = stage / 'source-native-package-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        site = stage / 'site'
        expected = {name: checksum for name, checksum in manifest['current_install_files'].items() if name.startswith('sgl_kernel/')}
        provenance = manifest_path
    assert os.environ['PYTHONPATH'] == str(site)
    actual = {str(path.relative_to(site)): digest(path) for path in (site / 'sgl_kernel').rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    assert actual == expected
    import torch
    import sgl_kernel

    assert Path(sgl_kernel.__file__).resolve() == site / 'sgl_kernel/__init__.py'
    assert str(Path(torch.__file__).resolve()).startswith('/opt/gfx-deps/venv/')
    assert not torch.xpu.is_initialized()
    for name in ['muse_spark_moe_routing', 'sgl_per_token_group_quant_fp4', 'sgl_padded_gather_per_token_group_quant_fp4', 'topk_sigmoid']:
        assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::' + name, 'XPU')
    mapped = sorted({Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/sgl_kernel/' in line and '.so' in line})
    assert mapped and any(path.name.startswith('common_ops.') for path in mapped)
    for path in mapped:
        assert path.is_relative_to(site) and digest(path) == expected[str(path.relative_to(site))], path
    result = {'kind': options.kind, 'package': str(sgl_kernel.__file__), 'site': str(site), 'package_files_sha256': expected,
              'mapped_native_sha256': {str(path): digest(path) for path in mapped}, 'provenance_sha256': digest(provenance),
              'Torch_provider': str(torch.__file__), 'Torch_version': torch.__version__, 'GPU_initialized': False,
              'scope': 'complete normal SOURCE package, historical GNU/icx or fresh tested-native icpx only; never official AFTER'}
    Path(options.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'kind': options.kind, 'package': result['package'], 'native_count': len(mapped), 'actual_exit': 0, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

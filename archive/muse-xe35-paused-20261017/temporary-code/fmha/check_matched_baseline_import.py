import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'preopt'), required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / 'matched-native-icpx-v1' / args.role
    manifest = json.loads((folder / 'normal-package-manifest.json').read_text())
    site = Path(manifest['site'])
    import torch
    assert not torch.xpu.is_initialized()
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve() == site / 'sgl_kernel/__init__.py'
    assert not torch.xpu.is_initialized()
    loaded = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith(str(site / 'sgl_kernel')):
            path = Path(fields[-1])
            loaded[str(path.relative_to(site))] = digest(path)
    for library, checksum in manifest['fresh_tested_native_libraries'].items():
        assert loaded[f'sgl_kernel/{library}'] == checksum
    for name, checksum in loaded.items():
        assert manifest['fresh_package_files_sha256'][name] == checksum
    schemas = [str(schema) for schema in torch._C._jit_get_all_schemas()
               if schema.name == 'sgl_kernel::fwd']
    assert schemas, 'real original normal registration missing; no fake op or ABI glue permitted'
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    report = {'role': args.role, 'package_file': sgl_kernel.__file__, 'gpu_initialized': torch.xpu.is_initialized(),
              'loaded_normal_package_libraries_sha256': loaded, 'original_fwd_schemas': schemas,
              'fresh_native_libraries': manifest['fresh_tested_native_libraries'],
              'normal_package_import': True, 'diagnostic_after_bridge': False, 'full_baseline_claimed': False}
    (folder / 'normal-import-verified.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'role': args.role, 'normal_import': True, 'GPU_initialized': False, 'loaded_DSOs': len(loaded)}))


if __name__ == '__main__':
    main()

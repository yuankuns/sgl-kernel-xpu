import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

from canonical_check import inventory, local_path


def audit(provenance, label, site):
    import sgl_kernel

    assert Path(sgl_kernel.__file__).resolve().is_relative_to(site.resolve())
    assert Path(sgl_kernel.common_ops.__file__).resolve().is_relative_to(site.resolve())
    for name, module in tuple(sys.modules.items()):
        if name == 'sgl_kernel' or name.startswith('sgl_kernel.'):
            origin = getattr(module, '__file__', None)
            if origin:
                assert Path(origin).resolve().is_relative_to(site.resolve()), (name, origin)
    if label == 'port':
        manifest_path = local_path(provenance['canonical_manifest'])
        assert inventory.digest_file(manifest_path) == provenance['canonical_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        expected = {name: record['sha256'] for name, record in (manifest['installed_files'] | manifest['python_snapshot_files']).items()}
    else:
        manifest_path = local_path(provenance[f'{label}_manifest'])
        assert inventory.digest_file(manifest_path) == provenance[f'{label}_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        if provenance.get(f'{label}_manifest_format') == 'relative-sha256-map':
            assert isinstance(manifest, dict)
            expected = manifest
        else:
            prefix = 'before/site' if label == 'preopt' else 'after/site'
            expected = {str(Path(record['path']).relative_to(prefix)): record['sha256'] for record in manifest}
    mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5])
        if not (path.name.startswith('libsgl-ops-') or path.name.startswith('common_ops.')):
            continue
        assert path.resolve().is_relative_to(site.resolve()), f'Foreign native library mapped: {path}'
        relative = str(path.relative_to(site))
        assert relative in expected
        mapped[str(path)] = inventory.digest_file(path)
        assert mapped[str(path)] == expected[relative], f'Mapped native hash mismatch: {path}'
    names = {Path(path).name for path in mapped}
    assert Path(sgl_kernel.common_ops.__file__).name in names
    assert 'libsgl-ops-sycl-FusedQKNormRope.so' in names
    if label == 'port':
        assert 'libsgl-ops-sycl-FusedQKNormRope_xe35.so' in names
        assert callable(sgl_kernel.fused_qk_norm_rope_xe35)
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--label', choices=('source', 'port', 'preopt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ['--']:
        command = command[1:]
    assert command
    assert os.environ['ZE_AFFINITY_MASK'] == '6', 'Official QK runtime is GPU6 only'
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    provenance = json.loads(arguments.provenance.read_text())
    site = local_path(provenance[f'{arguments.label}_site'])
    assert Path(os.environ['PYTHONPATH']).resolve() == site.resolve()
    import torch
    import sgl_kernel

    assert hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU unavailable'
    major, minor = torch.ops.sgl_kernel.query_device.default(torch.xpu.current_device())
    assert major == 3 and minor == 5, 'Xe35-only official cases require architecture 3.5'
    before = audit(provenance, arguments.label, site)
    actual_exit = 0
    if command[0] == '-c':
        assert len(command) == 2
        sys.argv = ['-c']
        exec(compile(command[1], '<official-normal-import>', 'exec'), {'__name__': '__main__'})
    elif command[:2] == ['-m', 'pytest']:
        sys.argv = ['pytest', *command[2:]]
        try:
            runpy.run_module('pytest', run_name='__main__')
        except SystemExit as result:
            if result.code not in (None, 0):
                raise
    else:
        script = Path(command[0]).resolve()
        sys.path.insert(0, str(script.parent))
        sys.argv = command
        try:
            runpy.run_path(str(script), run_name='__main__')
        except SystemExit as result:
            if result.code not in (None, 0):
                if arguments.label != 'preopt':
                    raise
                actual_exit = result.code
        except AssertionError:
            if arguments.label != 'preopt':
                raise
            traceback.print_exc()
            actual_exit = 1
    after = audit(provenance, arguments.label, site)
    torch.xpu.synchronize()
    inventory.write_json(arguments.output, {'label': arguments.label, 'normal_package_import_verified': True, 'all_runtime_mapped_dso_verified': True, 'site': str(site), 'device_arch': '3.5', 'gpu_affinity': os.environ['ZE_AFFINITY_MASK'], 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'before': before, 'after': after, 'package': sgl_kernel.__file__, 'common_ops': sgl_kernel.common_ops.__file__, 'provenance_sha256': hashlib.sha256(arguments.provenance.read_bytes()).hexdigest()})
    if actual_exit:
        raise SystemExit(actual_exit)


if __name__ == '__main__':
    main()

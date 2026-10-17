import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('native_support', ROOT / 'mxfp4-w4a4/fresh_source_native_v1.py')
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)


def stage_site(artifact, stage, stems):
    from wheel.wheelfile import WheelFile
    assert sys.executable == '/opt/gfx-deps/venv/bin/python'
    assert os.environ['LD_LIBRARY_PATH'].startswith('/opt/gfx-deps/profiling_tools/lib:')
    assert os.sched_getaffinity(0) == set(range(64, 80))
    assert int((stage / 'native-build.exit').read_text()) == 0
    original = next((artifact / 'source-wheels').glob('*.whl'))
    expected = json.loads((artifact / 'source-package-provenance.json').read_text())['wheel_sha256']
    assert SUPPORT.digest(original) == expected
    old_site = next((artifact / 'source-venv/lib').glob('python*/site-packages/sgl_kernel'))
    common = next(old_site.glob('common_ops*.so'))
    undefined = {line.split()[-1] for line in subprocess.check_output(['nm', '-D', '--undefined-only', str(common)], text=True).splitlines()}
    replacements = {}
    abi = {}
    for stem in stems:
        name = 'libsgl-ops-sycl-' + stem + '.so'
        native = stage / 'build/src' / name
        old = old_site / name
        exports = subprocess.check_output(['nm', '-D', '--defined-only', str(native)], text=True)
        previous = subprocess.check_output(['nm', '-D', '--defined-only', str(old)], text=True)
        required = sorted({line.split()[-1] for line in previous.splitlines()} & undefined)
        assert required and set(required) <= {line.split()[-1] for line in exports.splitlines()}, name
        dynamic = subprocess.check_output(['readelf', '-d', str(native)], text=True)
        assert 'Library soname: [' + name + ']' in dynamic
        replacements['sgl_kernel/' + name] = native
        abi[name] = {'required_original_common_ops_symbols': required, 'old_sha256': SUPPORT.digest(old), 'fresh_sha256': SUPPORT.digest(native)}
    wheels = stage / 'isolated-site-wheels'
    wheels.mkdir()
    fresh = wheels / original.name
    before = {}
    current = {}
    with zipfile.ZipFile(original) as package, WheelFile(fresh, 'w') as output:
        for name in package.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            content = package.read(name)
            before[name] = SUPPORT.hashlib.sha256(content).hexdigest()
            if name in replacements:
                assert before[name] == abi[Path(name).name]['old_sha256']
                content = replacements[name].read_bytes()
            current[name] = SUPPORT.hashlib.sha256(content).hexdigest()
            output.writestr(name, content)
    changed = [name for name in before if before[name] != current[name]]
    assert set(changed) == set(replacements)
    site = stage / 'site'
    assert not site.exists()
    SUPPORT.monitored(stage, 'site-package-install', [sys.executable, '-m', 'pip', 'install', '--no-deps',
                      '--no-build-isolation', '--target', str(site), str(fresh)], 120)
    for name, expected_hash in current.items():
        assert SUPPORT.digest(site / name) == expected_hash, name
    code = "import torch,sgl_kernel,pathlib,json,os; root=pathlib.Path(sgl_kernel.__file__).resolve(); print(json.dumps({'package':str(root),'torch':str(pathlib.Path(torch.__file__).resolve()),'torch_version':torch.__version__,'GPU_initialized':torch.xpu.is_initialized(),'mapped':sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line})})); assert str(root)==" + repr(str(site / 'sgl_kernel/__init__.py')) + "; assert str(pathlib.Path(torch.__file__).resolve()).startswith('/opt/gfx-deps/venv/'); assert not torch.xpu.is_initialized()"
    previous_path = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = str(site)
    try:
        SUPPORT.monitored(stage, 'site-package-import', [sys.executable, '-c', code], 120)
    finally:
        if previous_path is None:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = previous_path
    imported = json.loads((stage / 'site-package-import.log').read_text().splitlines()[-1])
    mapped = imported['mapped']
    assert str(site / 'sgl_kernel' / common.name) in mapped
    assert all(str(site / name) in mapped for name in replacements)
    SUPPORT.record(stage / 'source-native-package-manifest.json', {
        'scope': 'normal original complete SOURCE package installed with pip --target; only listed fresh original-source native replaced. Original registration/Python/other native unchanged. Not full/all-icpx SOURCE or official AFTER.',
        'original_wheel_sha256': SUPPORT.digest(original), 'fresh_wheel_sha256': SUPPORT.digest(fresh),
        'original_package_files': before, 'current_install_files': current, 'changed_package_files': changed,
        'ABI': abi, 'mapped_native_sha256': {name: SUPPORT.digest(name) for name in mapped},
        'normal_import_exit': 0, 'GPU_started': False, 'reference_precision_not_run': True,
        'provider': '/opt/gfx-deps/venv/bin/python', 'site_root': str(site), 'Torch_provider': imported['torch'],
        'Torch_version': imported['torch_version'], 'no_new_venv_or_Torch_copy': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True, choices=list(SUPPORT.FAMILIES))
    parser.add_argument('--stage-name', default='fresh-source-icpx-native-v2')
    options = parser.parse_args()
    artifact = ROOT / options.module
    stage = artifact / options.stage_name
    status = 1
    try:
        stage_site(artifact, stage, SUPPORT.FAMILIES[options.module][2])
        status = 0
    finally:
        (stage / 'isolated-site-stage.exit').write_text(str(status) + '\n')
    raise SystemExit(status)

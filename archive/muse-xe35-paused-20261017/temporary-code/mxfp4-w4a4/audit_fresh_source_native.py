import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True, choices=['mxfp4-w4a4', 'routing-quant'])
    parser.add_argument('--stage-name', default='fresh-source-icpx-native-v2')
    options = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    stage = root / options.module / options.stage_name
    stages = ['torch-cpu-import-preflight', 'configure', 'native-build', 'site-package-install',
              'site-package-import', 'isolated-site-stage']
    for name in stages:
        assert int((stage / (name + '.exit')).read_text()) == 0, name
    for name in ['torch-cpu-import-preflight', 'configure', 'native-build', 'site-package-install', 'site-package-import']:
        evidence = json.loads((stage / (name + '.resource-summary.json')).read_text())
        assert evidence['drained'] and evidence['guard'] is None, name
    post = json.loads((stage / 'source-postverify.json').read_text())
    assert all(post.values()), post
    policy = json.loads((stage / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM' and policy['SYCL'] == '/opt/gfx-deps/dpcpp/bin/icpx'
    manifest = json.loads((stage / 'source-native-package-manifest.json').read_text())
    assert manifest['GPU_started'] is False and manifest['normal_import_exit'] == 0
    site = stage / 'site'
    for name, expected in manifest['current_install_files'].items():
        assert digest(site / name) == expected, name
    actual_site = {str(path.relative_to(site)): digest(path) for path in (site / 'sgl_kernel').rglob('*')
                   if path.is_file() and '__pycache__' not in path.parts}
    expected_site = {name: expected for name, expected in manifest['current_install_files'].items() if name.startswith('sgl_kernel/')}
    assert actual_site == expected_site
    for path, expected in manifest['mapped_native_sha256'].items():
        host_path = Path(path) if Path(path).exists() else Path(path.replace('/root/', '/home/syk/', 1))
        assert digest(host_path) == expected, path
    natives = list((stage / 'build/src').glob('*.so'))
    assert {path.name for path in natives} == set(manifest['ABI']), 'No unlisted native/common_ops build allowed'
    objects = {}
    for path in sorted((stage / 'build/src').rglob('*.o')):
        sections = subprocess.check_output(['readelf', '-SW', str(path)], text=True)
        objects[str(path.relative_to(stage))] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                                                'offload_sections': [line.strip() for line in sections.splitlines() if 'offload' in line]}
    assert objects
    files = [path for path in stage.iterdir() if path.is_file() and path.suffix in ['.json', '.command', '.log', '.exit', '.patch', '.commands']]
    evidence_sha = {path.name: digest(path) for path in sorted(files) if path.name != 'cpu-native-acceptance.json'}
    report = {'module': options.module, 'actual_exit': 0, 'source_revision': json.loads((stage / 'source-preverify.json').read_text())['source_HEAD'],
              'dependency_pin': 'e87870bb7aacf83bfb3be9da685f148316b8df23', 'tracked_source_and_dependency_postverify': post,
              'compiler_policy': policy, 'tested_native_ABI': manifest['ABI'], 'objects': objects,
              'normal_complete_source_package_current_install_sha256': manifest['current_install_files'],
              'actual_mapped_native_sha256': manifest['mapped_native_sha256'], 'evidence_sha256': evidence_sha,
              'earlier_stage_true_exits_retained': {name: int((stage / (name + '.exit')).read_text()) for name in
                  ['create-private-venv', 'normal-package-install', 'normal-package-import', 'container', 'wrapper', 'supervisor']
                  if (stage / (name + '.exit')).exists()},
              'python_provider': manifest['provider'], 'Torch_provider': manifest['Torch_provider'],
              'full_current_site_files': {str(path.relative_to(site)): digest(path) for path in sorted(site.rglob('*'))
                  if path.is_file() and '__pycache__' not in path.parts},
              'limits': 'CPU-only finite listed SOURCE native rebuild/normal original registration package import. Not source whole-package fresh, not official AFTER/default-all, no GPU accuracy/performance run.'}
    (stage / 'cpu-native-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'module': options.module, 'actual_exit': 0, 'native_count': len(natives), 'object_count': len(objects), 'GPU_started': False}))


if __name__ == '__main__':
    main()

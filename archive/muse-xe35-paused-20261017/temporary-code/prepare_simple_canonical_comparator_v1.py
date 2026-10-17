import argparse
import json
import shutil
from pathlib import Path

from canonical_tools_loader_v1 import digest_file, inventory


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--build-version', choices=('canonical-v4-icpx',), default='canonical-v4-icpx')
    parser.add_argument('--validation-version', default='canonical-validation-v4-icpx')
    parser.add_argument('--source-native-manifest', type=Path, required=True)
    arguments = parser.parse_args()
    for version in (arguments.build_version, arguments.validation_version):
        if '/' in version or version in {'.', '..'}:
            raise SystemExit('Invalid evidence version')
    artifact = ROOT / arguments.component
    official_root = artifact / arguments.build_version
    official = json.loads((official_root / 'source-binary-manifest.json').read_text())
    source_manifest = arguments.source_native_manifest.resolve(strict=True)
    source_manifest.relative_to((artifact / 'fresh-source-icpx-v1/runs').resolve())
    source = json.loads(source_manifest.read_text())
    if official['build_evidence']['exit'] != 0 or official['install_actual_exit'] != 0:
        raise RuntimeError('Actual default-all and fresh current-graph install must succeed')
    if source.get('component') != arguments.component or source.get('tested_native_actual_exit') != 0:
        raise RuntimeError('Fresh SOURCE tested-native build must succeed')
    if source.get('source_snapshot', {}).get('all_kernel_bodies_helpers_preserved') is not True:
        raise RuntimeError('Original SOURCE kernel bodies and helpers must be preserved')
    attempt = source_manifest.parent
    for name in ('supervisor.exit', 'tested-native-build.exit'):
        if (attempt / name).read_text().strip() != '0':
            raise RuntimeError(f'Fresh SOURCE actual terminal exit is not successful: {name}')
    policy_path = attempt / 'compiler-policy-verified.json'
    if digest_file(policy_path) != source['compiler_policy_sha256']:
        raise RuntimeError('Changed SOURCE compiler policy')
    policy = json.loads(policy_path.read_text())
    for name in ('CMAKE_CXX_COMPILER', 'SYCL_COMPILER', 'SYCL_HOST_COMPILER'):
        if policy.get(name) != '/opt/gfx-deps/dpcpp/bin/icpx':
            raise RuntimeError(f'Fresh SOURCE compiler must be icpx: {name}')
    if policy.get('compiler_id') != 'IntelLLVM' or policy.get('source_flags_AOT_preserved') is not True:
        raise RuntimeError('Genuine SOURCE IntelLLVM compiler and preserved flags required')
    source_expected = {name: record['sha256'] for name, record in source['native'].items()}
    source_libraries = list((attempt / 'build').glob('libsgl-ops-sycl-*.so'))
    actual_source = {library.name: digest_file(library) for library in source_libraries}
    if actual_source != source_expected:
        raise RuntimeError('Source comparator differs from tested original fresh native kernels')
    official_site = official_root / 'site'
    expected_official = official['installed_files'] | official['python_snapshot_files']
    if inventory(official_site) != expected_official:
        raise RuntimeError('Official current-graph package has changed')
    validation = artifact / arguments.validation_version
    validation.mkdir(exist_ok=False)
    baseline = validation / 'before/site'
    shutil.copytree(official_site, baseline)
    for library in source_libraries:
        destination = baseline / 'sgl_kernel' / library.name
        if not destination.is_file():
            raise RuntimeError(f'Missing matching current native target: {destination}')
        shutil.copy2(library, destination)
    registration = official['common_ops']
    if digest_file(baseline / registration) != expected_official[registration]['sha256']:
        raise RuntimeError('Source comparator registration differs from current official registration')
    payload = {
        'component': arguments.component,
        'source_revision': source['source_revision'],
        'source_native_manifest': str(source_manifest),
        'source_native_manifest_sha256': digest_file(source_manifest),
        'source_compiler_policy': str(policy_path),
        'source_compiler_policy_sha256': digest_file(policy_path),
        'source_tested_native_compiler': '/opt/gfx-deps/dpcpp/bin/icpx',
        'official_base': official['source']['head'],
        'stage': str(baseline),
        'method': 'original optimized native kernels with tested native C++ symbol aliases, current normal official registration; SOURCE comparator only',
        'original_native_libraries': actual_source,
        'registration': registration,
        'files': inventory(baseline),
        'after_site': str(official_site),
        'after_manifest': str(official_root / 'source-binary-manifest.json'),
        'after_manifest_sha256': digest_file(official_root / 'source-binary-manifest.json'),
        'after_modified': False,
    }
    (validation / 'source-comparator-manifest.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'component': arguments.component, 'source_comparator': str(baseline), 'after_site': str(official_site), 'after_modified': False}))


if __name__ == '__main__':
    main()

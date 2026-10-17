import argparse
import hashlib
import json
from pathlib import Path


ART = Path(__file__).resolve().parent.parent
ROOT = ART / 'matched-native-icpx-v1'
HEADS = {'source': '08955d90fad52a40779705f259569db93e570a22',
         'preopt': 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def local_path(path):
    if str(ART).startswith('/home/syk/'):
        return Path(str(path).replace('/root/', '/home/syk/', 1))
    return Path(path)


def provenance(role):
    assert role in HEADS
    folder = ROOT / role
    assert (folder / 'fresh-native-binaries.json').is_file()
    fresh_record = json.loads((folder / 'fresh-native-binaries.json').read_text())
    terminal = json.loads((folder / fresh_record.get('terminal_file', 'native-build.terminal.json')).read_text())
    assert terminal['exit'] == 0 and terminal['remaining_known_live'] == []
    assert json.loads((folder / 'inputs-prebuild.json').read_text()) == json.loads((folder / 'inputs-postbuild.json').read_text())
    manifest_path = folder / 'normal-package-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_head'] == HEADS[role]
    assert manifest['default_all_baseline_claimed'] is False
    assert manifest['diagnostic_abi_bridge'] is False
    package = local_path(manifest['site']) / 'sgl_kernel'
    for name, checksum in manifest['fresh_package_files_sha256'].items():
        assert digest(package.parent / name) == checksum, name
    imported = json.loads((folder / 'normal-import-verified.json').read_text())
    assert imported['normal_package_import'] is True and imported['gpu_initialized'] is False
    graph = json.loads((folder / 'compiler-graph-verified.json').read_text())
    assert 'set(CMAKE_CXX_COMPILER_ID "IntelLLVM")' in graph['cxx_identity']
    assert 'set(CMAKE_C_COMPILER "/opt/gfx-deps/dpcpp/bin/icx")' in graph['c_identity']
    ninja_log = (folder / 'build/.ninja_log').read_text()
    kernels = {}
    for target, record in graph['targets'].items():
        library = f'lib{target}.so'
        assert f'src/{library}' in ninja_log
        checksum = manifest['fresh_tested_native_libraries'][library]
        assert digest(folder / 'build/src' / library) == checksum
        assert digest(package / library) == imported['fresh_native_libraries'][library] == checksum
        assert digest(local_path(record['generated_command'])) == record['generated_sha256']
        assert 'set(SYCL_executable "/opt/gfx-deps/dpcpp/bin/icpx")' in record['native_recipe']
        assert 'set(SYCL_host_compiler "/opt/gfx-deps/dpcpp/bin/icpx")' in record['native_recipe']
        kernels[target] = {'library': library, 'library_sha256': checksum, 'native_sycl_driver': '/opt/gfx-deps/dpcpp/bin/icpx',
                          'native_invocation_basename_is_icpx': True, 'sycl_host_compiler': '/opt/gfx-deps/dpcpp/bin/icpx', **record}
    return {'provider': role, 'source_head': HEADS[role], 'native_kernels': kernels,
            'cxx_compiler': '/opt/gfx-deps/dpcpp/bin/icpx', 'cxx_compiler_id': 'IntelLLVM',
            'compiler_identity': graph['cxx_identity'], 'c_compiler_identity': graph['c_identity'],
            'package_manifest': str(manifest_path), 'package_manifest_sha256': digest(manifest_path),
            'site': manifest['site'], 'completed_ninja_log_sha256': digest(folder / 'build/.ninja_log'),
            'actual_native_dependency_manifest_sha256': digest(folder / 'actual-native-dependencies.json'),
            'cutlass_historical_label': 'e87870bb7aacf83bfb3be9da685f148316b8df23',
            'full_e878_pre_post_verified_files': 5874, 'tested_native_cross_compiler_confound': False,
            'cross_compiler_performance_confound': True,
            'confound': 'tested three native drivers/hosts/links matched icpx; non-target package components retain historical GNU/icx; e878/profiling ON versus official public7b56/profiling OFF remain different',
            'evidence_boundary': 'three fresh original native targets, actual generated recipes/Ninja completion/normal registered package maps; not baseline default-all'}


def package_preflight(role):
    report = provenance(role)
    manifest = json.loads(Path(report['package_manifest']).read_text())
    package = local_path(report['site']) / 'sgl_kernel'
    hashes = {Path(name).name: checksum for name, checksum in manifest['fresh_package_files_sha256'].items()}
    return package, hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'preopt'), required=True)
    parser.add_argument('--site-container-only', action='store_true')
    args = parser.parse_args()
    report = provenance(args.role)
    print(report['site'] if args.site_container_only else json.dumps(report, indent=2))

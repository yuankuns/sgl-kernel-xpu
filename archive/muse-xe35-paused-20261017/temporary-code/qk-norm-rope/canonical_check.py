import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ARTIFACT = Path(__file__).resolve().parent
TARGET = ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35'
CURRENT_VERSION = 'canonical-v4-icpx'
CANONICAL = ARTIFACT / os.environ.get('QK_CANONICAL_VERSION', CURRENT_VERSION)
READY = ARTIFACT / 'official-source-ready.json'
PIN = '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
specification = importlib.util.spec_from_file_location('qk_snapshot_inventory', ARTIFACT.parent / 'canonical-tools/overlay_snapshot.py')
inventory = importlib.util.module_from_spec(specification)
specification.loader.exec_module(inventory)


def local_path(value):
    prefix, replacement = ('/root/', '/home/syk/') if str(ARTIFACT).startswith('/home/syk/') else ('/home/syk/', '/root/')
    return Path(value.replace(prefix, replacement, 1) if value.startswith(prefix) else value)


def scoped_git_output(repository, *arguments):
    repository = str(Path(repository).resolve())
    return subprocess.check_output(['git', '-c', f'safe.directory={repository}', '--no-optional-locks', '-C', repository, *arguments])


def freeze():
    assert scoped_git_output(TARGET, 'rev-parse', 'HEAD').decode().strip() == inventory.OFFICIAL_BASE
    files, selection = inventory.source_inventory(TARGET)
    payload = {'schema': 1, 'created_epoch': time.time(), 'source': str(TARGET), 'head': inventory.OFFICIAL_BASE, 'official_base': inventory.OFFICIAL_BASE, 'branch': scoped_git_output(TARGET, 'branch', '--show-current').decode().strip(), 'source_digest': inventory.manifest_digest(files), 'files': files, 'selection': selection, 'git_status': scoped_git_output(TARGET, 'status', '--porcelain').decode(), 'ready_scope': 'frozen PR source only; canonical full package and GPU6 official verification pending'}
    if READY.exists():
        old = json.loads(READY.read_text())
        assert old['source_digest'] == payload['source_digest'] and old['files'] == files, 'Frozen source changed; explicit new freeze/version required'
        return old
    inventory.write_json(READY, payload)
    return payload


def canonical_directory(version):
    assert version == CURRENT_VERSION, 'Old versions are diagnostics; require reviewed canonical-v4-icpx'
    return ARTIFACT / version


def verify_protected_qk_source(frozen):
    original = json.loads(READY.read_text())
    protected = ('src/sycl/xe35/FusedQKNormRope.cpp', 'src/sycl/xe35/FusedQKNormRopeKernel.h', 'src/sycl/xe35/FusedQKNormRopeFrequency.h', 'src/sycl/FusedQKNormRope.cpp', 'include/sgl_kernel/xe35/fused_qk_norm_rope.h', 'include/sgl_kernel_ops.h', 'src/torch_extension_sycl.cc', 'python/sgl_kernel/__init__.py', 'python/sgl_kernel/elementwise.py', 'tests/test_fused_qk_norm_rope.py', 'benchmark/muse_scope.py', 'benchmark/bench_muse_qk_norm_rope.py', 'benchmark/validate_qk_norm_rope_b8.py', 'src/sycl/MemoryAccess.h', 'src/sycl/Norm.h', 'src/sycl/SYCLHelpers.h', 'src/sycl/Utils.h')
    for name in protected:
        assert frozen['files'][name] == original['files'][name], f'Protected QK algorithm/API/reference changed: {name}'
    return protected


def verify_reviewed_freeze(version=CURRENT_VERSION):
    canonical_directory(version)
    reviewed_path = ARTIFACT / 'canonical-source-freeze-v4-icpx.json'
    reviewed = json.loads(reviewed_path.read_text())
    assert reviewed['version'] == version and reviewed['head'] == inventory.OFFICIAL_BASE
    assert local_path(reviewed['source']).resolve() == TARGET.resolve()
    assert reviewed['source_digest'] == inventory.manifest_digest(reviewed['files'])
    prior = json.loads((ARTIFACT / 'canonical-v2/source-snapshot.json').read_text())
    v3 = json.loads((ARTIFACT / 'canonical-source-freeze-v3-icpx.json').read_text())
    assert v3['version'] == 'canonical-v3-icpx'
    assert v3['source_digest'] == inventory.manifest_digest(v3['files'])
    assert set(prior['files']) == set(v3['files']) == set(reviewed['files'])
    assert {name for name in prior['files'] if prior['files'][name] != v3['files'][name]} == {'cmake/BuildFlags.cmake'}
    assert v3['changed_paths_from_v2'] == ['cmake/BuildFlags.cmake']
    assert {name for name in v3['files'] if v3['files'][name] != reviewed['files'][name]} == {'src/sycl/HCHead.cpp'}
    assert reviewed['changed_paths_from_v3'] == ['src/sycl/HCHead.cpp']
    original_flags = scoped_git_output(TARGET, 'show', f'{inventory.OFFICIAL_BASE}:cmake/BuildFlags.cmake')
    assert hashlib.sha256(original_flags).hexdigest() == prior['files']['cmake/BuildFlags.cmake']['sha256']
    expected_flags = original_flags.replace(b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")', b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")').replace(b'Currently only support GCC compiler on Linux as CXX compiler.', b'Currently only support GNU or IntelLLVM compiler on Linux as CXX compiler.')
    assert hashlib.sha256(expected_flags).hexdigest() == v3['files']['cmake/BuildFlags.cmake']['sha256']
    assert (TARGET / 'cmake/BuildFlags.cmake').read_bytes() == expected_flags, 'Only reviewed two-line compiler gate/message change allowed'
    original_head = scoped_git_output(TARGET, 'show', f'{inventory.OFFICIAL_BASE}:src/sycl/HCHead.cpp')
    assert hashlib.sha256(original_head).hexdigest() == v3['files']['src/sycl/HCHead.cpp']['sha256']
    attributes = b'  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]] void\n'
    assert original_head.count(attributes) == 1
    expected_head = original_head.replace(attributes, b'#ifdef __SYCL_DEVICE_ONLY__\n  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]]\n#endif\n  void\n')
    assert (TARGET / 'src/sycl/HCHead.cpp').read_bytes() == expected_head, 'Only reviewed device-only unchanged attribute guard allowed'
    proof = json.loads((ARTIFACT.parent / 'ICPX_COMPAT_SOURCE_FREEZE_AUDIT_v4.json').read_text())
    assert proof['version'] == version and proof['original_device_ir_after_source_path_normalization_identical'] is True
    assert proof['no_math_or_launch_dimension_change'] is True
    assert proof['proof_sha256'] == reviewed['device_ir_proof_sha256']
    verify_protected_qk_source(reviewed)
    actual, selection = inventory.source_inventory(TARGET)
    assert actual == reviewed['files'] and selection == reviewed['selection'], 'Actual PR differs from main reviewed icpx freeze'
    return reviewed, reviewed_path


def verify_main_ready(version, manifest_path=None):
    directory = canonical_directory(version)
    path = local_path(str(manifest_path)) if manifest_path else directory / 'source-snapshot.json'
    frozen = json.loads(path.read_text())
    assert frozen['verified'] and frozen['head'] == frozen['official_base'] == inventory.OFFICIAL_BASE
    assert local_path(frozen['source']).resolve() == TARGET.resolve()
    assert frozen['source_digest'] == inventory.manifest_digest(frozen['files'])
    reviewed, reviewed_path = verify_reviewed_freeze(version)
    assert reviewed['source_digest'] == frozen['source_digest']
    assert reviewed['files'] == frozen['files']
    verify_protected_qk_source(frozen)
    return frozen, path


def verify_default_command(text, build):
    command = shlex.split(text)
    assert command and Path(command[0]).name == 'cmake', 'Require actual cmake default-all build command'
    assert len(command) >= 3 and command[1] == '--build' and local_path(command[2]).resolve() == build.resolve()
    remainder = command[3:]
    while remainder:
        option = remainder.pop(0)
        assert option in ('--parallel', '-j', '--config'), f'Targeted/non-default build option rejected: {option}'
        assert remainder, 'Missing build option value'
        value = remainder.pop(0)
        assert value == 'Release' if option == '--config' else value.isdigit() and 1 <= int(value) <= 56


def verify_evidence(evidence, build):
    assert evidence['exit'] == 0
    for kind in ('exit', 'command', 'log'):
        path = local_path(evidence[f'{kind}_file'])
        assert inventory.digest_file(path) == evidence[f'{kind}_file_sha256' if kind == 'exit' else f'{kind}_sha256']
    assert local_path(evidence['exit_file']).read_text().strip() == '0'
    assert local_path(evidence['command_file']).read_text() == evidence['command_text']
    verify_default_command(evidence['command_text'], build)


def check(version=CURRENT_VERSION, ready_manifest=None):
    canonical = canonical_directory(version)
    frozen, ready_path = verify_main_ready(version, ready_manifest)
    manifest_path = canonical / 'source-binary-manifest.json'
    payload = json.loads(manifest_path.read_text())
    from compiler_policy_check import verify_compiler_policy

    compiler_proof = verify_compiler_policy(canonical, payload, local_path, inventory)
    release_policy = json.loads((ARTIFACT / 'official-release-policy.json').read_text())
    assert version == release_policy['next_actual_package_version']
    assert frozen['source_digest'] == release_policy['required_source_digest']
    for name in ('configure', 'default-all-build', 'supervisor'):
        assert (canonical / f'{name}.exit').read_text().strip() == '0'
    post = json.loads((canonical / 'source-readonly-postverify.json').read_text())
    assert post['verified'] and local_path(post['source']).resolve() == TARGET.resolve()
    assert local_path(post['snapshot_manifest']).resolve() == (canonical / 'source-snapshot.json').resolve()
    assert post['epoch'] >= payload['created_epoch']
    complete = json.loads((canonical / 'build-stage-complete.json').read_text())
    assert complete['default_all_exit'] == 0 and complete['fresh_current_graph_install'] is True
    assert complete['source_digest'] == frozen['source_digest'] and complete['component'] == 'qk-norm-rope'

    assert payload['schema'] == 1 and payload['install_actual_exit'] == 0
    assert payload['cutlass_pin'] == PIN and payload['default_features'] == FEATURES
    for name, value in FEATURES.items():
        assert payload['flags'][name]['value'] == value
    assert payload['flags']['DPCPP_SYCL_TARGET']['value'] == 'cri'
    assert payload['flags']['CMAKE_BUILD_TYPE']['value'] == 'Release'
    for name, record in payload['flags'].items():
        if name.startswith('CMAKE_') and 'FLAGS' in name:
            assert not re.search(r'(?:^|\s)-D\s*__INTEL_LLVM_COMPILER(?:=|\s|$)', record['value']), 'Global compiler identity spoof rejected'
    for name in ('CMAKE_PROJECT_INCLUDE', 'CMAKE_PROJECT_INCLUDE_BEFORE', 'CMAKE_PROJECT_TOP_LEVEL_INCLUDES'):
        assert not payload['flags'].get(name, {}).get('value')
    dependency = local_path(payload['flags']['FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL']['value'])
    assert scoped_git_output(dependency, 'rev-parse', 'HEAD').decode().strip() == PIN
    assert not scoped_git_output(dependency, 'status', '--porcelain').strip()
    source = payload['source']
    assert source['verified'] and source['head'] == source['official_base'] == inventory.OFFICIAL_BASE
    assert local_path(source['source']).resolve() == TARGET.resolve()
    assert source['files'] == frozen['files']
    assert source['source_digest'] == inventory.manifest_digest(source['files']) == frozen['source_digest']
    snapshot_manifest = canonical / 'source-snapshot.json'
    assert inventory.digest_file(snapshot_manifest) == payload['source_manifest_file_sha256']
    assert json.loads(snapshot_manifest.read_text()) == source
    site = local_path(payload['site'])
    assert site.resolve() == (canonical / 'site').resolve() and not site.is_symlink()
    build = local_path(payload['build'])
    assert local_path(payload['flags']['CMAKE_CACHEFILE_DIR']['value']).resolve() == build.resolve()
    assert local_path(payload['flags']['CMAKE_HOME_DIRECTORY']['value']).resolve() == local_path(payload['snapshot']).resolve()
    assert local_path(source['destination']).resolve() == local_path(payload['snapshot']).resolve()
    verify_evidence(payload['build_evidence'], build)
    install = payload['cmake_install_command']
    assert Path(install[0]).name == 'cmake' and install[1] == '--install'
    assert local_path(install[2]).resolve() == build.resolve()
    assert install[3] == '--prefix' and local_path(install[4]).resolve() == site.resolve()
    assert install[5:] == ['--config', 'Release']
    install_logs = [path for path in canonical.glob('*.log') if inventory.digest_file(path) == payload['install_log_sha256']]
    assert install_logs, 'Actual CMake install log missing or hash mismatch'
    assert 'CANONICAL_INSTALL_ACTUAL_EXIT=0' in install_logs[0].read_text()
    installed = payload['installed_files']
    python_files = {str(Path(name).relative_to('python')): record for name, record in source['files'].items() if name.startswith('python/sgl_kernel/')}
    assert python_files == payload['python_snapshot_files']
    inventory.verify_inventory(site, installed | python_files)
    install_paths = {str(local_path(line).relative_to(site)) for line in payload['install_manifest_text'].splitlines() if line}
    assert install_paths == set(installed)
    assert hashlib.sha256(payload['install_manifest_text'].encode()).hexdigest() == payload['install_manifest_sha256']
    native = {Path(name).name: name for name in installed if name.endswith('.so')}
    assert len(native) == sum(name.endswith('.so') for name in installed)
    assert set(native) == set(payload['binaries']) == set(payload['current_native_targets'])
    for name, relative in native.items():
        assert payload['binaries'][name] == {'path': relative, **installed[relative]}
    common = [name for name in native if name.startswith('common_ops.')]
    assert len(common) == 1 and payload['common_ops'] == native[common[0]]
    generic_name = 'libsgl-ops-sycl-FusedQKNormRope.so'
    kernel_name = 'libsgl-ops-sycl-FusedQKNormRope_xe35.so'
    assert generic_name in native and kernel_name in native
    assert installed[native[generic_name]]['sha256'] != 'ff605a2a1a1f7d25aef091bdaad6b42b3e6428284b703155e40c968d7e76a0d3', 'Diagnostic ABI bridge prohibited'
    visibility = {}
    for name, function in ((generic_name, 'fused_qk_norm_rope'), (kernel_name, 'fused_qk_norm_rope_xe35')):
        output = subprocess.check_output(['nm', '-D', '-C', str(site / native[name])], text=True)
        public = [line for line in output.splitlines() if f' T at::native::xpu::{function}(' in line]
        assert len(public) == 1, f'Native entry hidden: {name}'
        visibility[name] = public[0]
    dynamic = subprocess.check_output(['readelf', '-d', str(site / native[generic_name])], text=True)
    assert kernel_name in re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic), 'Generic ABI must link real Xe35 implementation'
    dynamic = subprocess.check_output(['readelf', '-d', str(site / payload['common_ops'])], text=True)
    needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)
    assert needed == payload['common_ops_dt_needed']
    assert all(name in native for name in needed if name.startswith('libsgl-ops-'))
    source_evidence = ARTIFACT.parents[2] / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
    subprocess.run([sys.executable, str(ARTIFACT / 'verify_source.py')], check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    baseline = json.loads((ARTIFACT / 'source-verification.json').read_text())
    source_manifest = source_evidence / 'after/site-manifest.json'
    result = {'mode': 'canonical-default-all-current-graph-fresh-install', 'source_commit': baseline['source_commit'], 'target_base': inventory.OFFICIAL_BASE, 'source_site': str(source_evidence / 'after/site'), 'source_manifest': str(source_manifest), 'source_manifest_sha256': inventory.digest_file(source_manifest), 'source_qk_library_sha256': baseline['source_qk_binary_sha256'], 'port_site': str(site), 'port_qk_library_sha256': installed[native[kernel_name]]['sha256'], 'canonical_manifest': str(manifest_path), 'canonical_manifest_sha256': inventory.digest_file(manifest_path), 'ready_source_sha256': frozen['source_digest'], 'port_hashes': {name: record['sha256'] for name, record in frozen['files'].items()}, 'default_features': FEATURES, 'public_cutlass_pin': PIN, 'visibility': visibility, 'default_all_command_verified': True, 'normal_package_import_verified': False, 'all_runtime_mapped_dso_verified': False, 'accuracy_verified': False, 'performance_verified': False, 'official_full_package_pass': False, 'build_provenance': 'coordinator persistent CMake tree incremental default-all; not an independent clean build', 'source_site_known_difference': baseline['known_full_package_source_difference']}
    result.update({'canonical_version': version, 'ready_manifest': str(ready_path), 'ready_manifest_sha256': inventory.digest_file(ready_path), 'original_qk_source_digest': json.loads(READY.read_text())['source_digest'], 'reviewed_source_freeze': str(ARTIFACT / 'canonical-source-freeze-v4-icpx.json'), 'compiler_policy_proof': compiler_proof, 'source_readonly_postverified': True, 'install_actual_exit': payload['install_actual_exit']})
    from fresh_native_check import verify_provider

    result['historical_source_qk_library_sha256'] = result['source_qk_library_sha256']
    result.update(verify_provider('optimized-source'))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--canonical-version', choices=(CURRENT_VERSION,), default=os.environ.get('QK_CANONICAL_VERSION', CURRENT_VERSION))
    parser.add_argument('--ready-manifest', type=Path)
    arguments = parser.parse_args()
    if arguments.freeze:
        result = freeze()
        print(json.dumps({'ready_source_sha256': result['source_digest'], 'base': result['official_base'], 'files': len(result['files']), 'official_package_pass': False}))
        return
    directory = canonical_directory(arguments.canonical_version)
    result = check(arguments.canonical_version, arguments.ready_manifest)
    if arguments.runtime_root:
        assert result['compiler_policy_proof']['verified'] and result['source_readonly_postverified']
        runtime_root = arguments.runtime_root.resolve()
        assert runtime_root.parent == directory.resolve() and runtime_root.name.startswith('official-')
        assert not runtime_root.exists(), 'Use a new versioned official run; never overwrite evidence'
        inventory.write_json(runtime_root / 'provenance.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

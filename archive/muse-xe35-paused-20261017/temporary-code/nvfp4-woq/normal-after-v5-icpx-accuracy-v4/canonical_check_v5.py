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
ARTIFACT = Path(__file__).resolve().parent.parent
sys.path.append(str(ARTIFACT))
TARGET = ARTIFACT.parents[1] / 'pr/muse-nvfp4-woq-xe35'
DEFAULT_VERSION = 'canonical-v5-icpx'
VERSION_POOLS = {'canonical-v5-icpx': 'canonical-icpx-pool-v4'}
CANONICAL = ARTIFACT / DEFAULT_VERSION
READY = ARTIFACT / f'canonical-source-freeze-{DEFAULT_VERSION.removeprefix("canonical-")}.json'
PIN = '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
specification = importlib.util.spec_from_file_location('nvfp4_snapshot_inventory', ARTIFACT.parent / 'canonical-tools/overlay_snapshot.py')
inventory = importlib.util.module_from_spec(specification)
specification.loader.exec_module(inventory)


def local_path(value):
    prefix, replacement = ('/root/', '/home/syk/') if str(ARTIFACT).startswith('/home/syk/') else ('/home/syk/', '/root/')
    return Path(value.replace(prefix, replacement, 1) if value.startswith(prefix) else value)


def select_version(version):
    global CANONICAL, READY
    if version not in VERSION_POOLS:
        raise RuntimeError('Normal AFTER requires reviewed canonical-v5-icpx; old versions/diagnostic ABI prohibited')
    CANONICAL = ARTIFACT / version
    READY = ARTIFACT / f'canonical-source-freeze-{version.removeprefix("canonical-")}.json'


def freeze(version=DEFAULT_VERSION):
    select_version(version)
    assert inventory.git_output(TARGET, 'rev-parse', 'HEAD').decode().strip() == inventory.OFFICIAL_BASE
    files, selection = inventory.source_inventory(TARGET)
    if not READY.is_file():
        raise RuntimeError(f'Main-owned versioned source freeze required: {READY}')
    payload = json.loads(READY.read_text())
    assert payload['head'] == inventory.OFFICIAL_BASE
    assert payload['source_digest'] == 'dd2adcfa688cd0df1cc154fbbd2ffa9105c53309f154b7aa3a7275cd6edd97a2'
    assert local_path(payload['source']).resolve() == TARGET.resolve()
    assert payload['files'] == files and payload['source_digest'] == inventory.manifest_digest(files), 'Main frozen source differs; stop queue and request a new main freeze'
    return {**payload, 'official_base': inventory.OFFICIAL_BASE, 'canonical_version': version}


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


def verify_compiler_policy(payload, directory):
    policy_path = directory / 'compiler-policy-verified.json'
    policy = json.loads(policy_path.read_text())
    cxx = '/opt/gfx-deps/dpcpp/bin/icpx'
    compiler_fields = {'CMAKE_CXX_COMPILER': cxx, 'CMAKE_C_COMPILER': '/opt/gfx-deps/dpcpp/bin/icx', 'SYCL_COMPILER': cxx, 'SYCL_HOST_COMPILER': cxx, 'XPU_SYCL_COMPILER': 'icx'}
    expected = {'compiler_id': 'IntelLLVM', 'cxx': cxx, 'c': compiler_fields['CMAKE_C_COMPILER'], 'sycl': cxx, 'host': cxx, 'torch_SYCL_family': 'icx', 'no_GNU_command': True}
    assert all(policy.get(name) == value for name, value in expected.items()), 'Real IntelLLVM CXX/SYCL/host icpx and C icx policy required'
    required = {'-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen'}
    assert required.issubset(policy['required_flags']), 'Production precision compiler policy incomplete'
    cache_path = directory / 'CMakeCache.txt'
    ninja_path = directory / 'build.ninja'
    assert inventory.digest_file(cache_path) == payload['cache_sha256']
    assert inventory.digest_file(ninja_path) == payload['ninja_sha256']
    cache = dict(re.findall(r'^([^#/:\n][^:\n]*):[^=\n]+=(.*)$', cache_path.read_text(), re.MULTILINE))
    for name, value in compiler_fields.items():
        assert payload['flags'][name]['value'] == cache.get(name) == value, f'Actual compiler cache mismatch: {name}'
    commands = ninja_path.read_text()
    for forbidden in ('/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc', '-fsycl-host-compiler=/usr/bin/'):
        assert forbidden not in commands, f'GNU host command rejected: {forbidden}'
    nv_commands = [line for line in commands.splitlines() if line.strip().startswith('COMMAND =') and 'sgl-ops-sycl-blockwise_moe_nvfp4_woq_sycl_device_obj.o' in line]
    assert len(nv_commands) == 1
    assert cxx + ' ' in nv_commands[0] and '-cl-intel-512-GRF-per-thread' in nv_commands[0], 'Source-fidelity GRF512 actual icpx AOT command required'
    return inventory.digest_file(policy_path)




def check(version=DEFAULT_VERSION):
    frozen = freeze(version)
    manifest_path = CANONICAL / 'source-binary-manifest.json'
    payload = json.loads(manifest_path.read_text())
    assert payload['schema'] == 1 and payload['install_actual_exit'] == 0
    assert payload['cutlass_pin'] == PIN and payload['default_features'] == FEATURES
    for name, value in FEATURES.items():
        assert payload['flags'][name]['value'] == value
    assert payload['flags']['DPCPP_SYCL_TARGET']['value'] == 'cri'
    assert payload['flags']['CMAKE_BUILD_TYPE']['value'] == 'Release'
    for name in ('CMAKE_PROJECT_INCLUDE', 'CMAKE_PROJECT_INCLUDE_BEFORE', 'CMAKE_PROJECT_TOP_LEVEL_INCLUDES'):
        assert not payload['flags'].get(name, {}).get('value')
    dependency = local_path(payload['flags']['FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL']['value'])
    assert inventory.git_output(dependency, 'rev-parse', 'HEAD').decode().strip() == PIN
    assert not inventory.git_output(dependency, 'status', '--porcelain').strip()
    source = payload['source']
    assert source['verified'] and source['head'] == source['official_base'] == inventory.OFFICIAL_BASE
    assert local_path(source['source']).resolve() == TARGET.resolve()
    assert source['files'] == frozen['files']
    assert source['source_digest'] == inventory.manifest_digest(source['files']) == frozen['source_digest']
    snapshot_manifest = CANONICAL / 'source-snapshot.json'
    assert inventory.digest_file(snapshot_manifest) == payload['source_manifest_file_sha256']
    assert json.loads(snapshot_manifest.read_text()) == source
    post = json.loads((CANONICAL / 'source-readonly-postverify.json').read_text())
    assert post['verified'] is True
    assert local_path(post['source']).resolve() == TARGET.resolve()
    assert local_path(post['snapshot_manifest']).resolve() == snapshot_manifest.resolve()
    complete = json.loads((CANONICAL / 'build-stage-complete.json').read_text())
    assert complete['component'] == 'nvfp4-woq' and complete['default_all_exit'] == 0
    assert complete['fresh_current_graph_install'] is True and complete['source_digest'] == frozen['source_digest']
    assert (CANONICAL / 'supervisor.exit').read_text().strip() == '0', 'Actual canonical supervisor must finish successfully'
    compiler_policy_sha256 = verify_compiler_policy(payload, CANONICAL)
    site = local_path(payload['site'])
    assert site.resolve() == (CANONICAL / 'site').resolve() and not site.is_symlink()
    build = local_path(payload['build'])
    pool = ARTIFACT.parent / VERSION_POOLS[version]
    assert build.resolve() == (pool / 'build').resolve(), 'Wrong canonical compiler pool/version'
    assert local_path(payload['snapshot']).resolve() == (pool / 'source').resolve()
    assert local_path(payload['flags']['CMAKE_CACHEFILE_DIR']['value']).resolve() == build.resolve()
    assert local_path(payload['flags']['CMAKE_HOME_DIRECTORY']['value']).resolve() == local_path(payload['snapshot']).resolve()
    assert local_path(source['destination']).resolve() == local_path(payload['snapshot']).resolve()
    verify_evidence(payload['build_evidence'], build)
    install = payload['cmake_install_command']
    assert Path(install[0]).name == 'cmake' and install[1] == '--install'
    assert local_path(install[2]).resolve() == build.resolve()
    assert install[3] == '--prefix' and local_path(install[4]).resolve() == site.resolve()
    assert install[5:] == ['--config', 'Release']
    install_logs = [path for path in CANONICAL.glob('*.log') if inventory.digest_file(path) == payload['install_log_sha256']]
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
    required = {
        'libsgl-ops-sycl-nvfp4_woq.so': ('nvfp4_woq_prep_xe35',),
        'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so': ('nvfp4_woq_bf16_grouped_mm_xe35', 'nvfp4_woq_mxfp8_scaled_grouped_mm_xe35'),
        'libsgl-ops-sycl-nvfp4_woq_scale_layout.so': ('mxfp4_moe_scale_to_mn_major_xe35',),
    }
    visibility = {}
    for name, functions in required.items():
        assert name in native, f'Missing actual NVFP4 native library: {name}'
        symbols = subprocess.check_output(['nm', '-D', '-C', '--defined-only', str(site / native[name])], text=True)
        visibility[name] = []
        for function in functions:
            matches = [line for line in symbols.splitlines() if ' T ' in line and function + '(' in line]
            assert len(matches) == 1, f'Hidden/duplicate NVFP4 native entry: {function}'
            visibility[name].append(matches[0])
    ninja = CANONICAL / 'build.ninja'
    assert inventory.digest_file(ninja) == payload['ninja_sha256']
    commands = [line for line in ninja.read_text().splitlines() if line.strip().startswith('COMMAND =') and 'sgl-ops-sycl-blockwise_moe_nvfp4_woq_sycl_device_obj.o' in line and '/icpx ' in line]
    assert len(commands) == 1 and '-cl-intel-512-GRF-per-thread' in commands[0], 'Source GRF512 policy absent in archived actual AOT command'
    cache = CANONICAL / 'CMakeCache.txt'
    assert inventory.digest_file(cache) == payload['cache_sha256']
    assert not any('_port_payload' in name for name in native), 'Diagnostic payload prohibited'
    dynamic = subprocess.check_output(['readelf', '-d', str(site / payload['common_ops'])], text=True)
    needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', dynamic)
    assert needed == payload['common_ops_dt_needed']
    assert all(name in native for name in needed if name.startswith('libsgl-ops-'))
    baseline = ARTIFACT / 'native-diagnostic/source/site/sgl_kernel'
    expected_source = {
        'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so': '864f695ee11647af36ea1140c941d3ce6f68ae3c1c22a5928784d34766ff4e06',
        'libsgl-ops-sycl-nvfp4_woq.so': 'bad2595a97d75d903fecac9770ddc2bf659baaee1cff362c84204108b6c131fa',
        'libsgl-ops-sycl-mxfp4_moe_scale_layout.so': '2196b48e0e601bd78f9610bc2b6f5541c8942ba7b5e78d30af2946acef33e3d4',
    }
    for name, digest in expected_source.items():
        assert inventory.digest_file(baseline / name) == digest, f'd983e87 source native SHA changed: {name}'
    registrations = list(baseline.glob('common_ops*.so'))
    assert len(registrations) == 1
    assert inventory.digest_file(registrations[0]) == 'e63d50179a696799f4e6b57bb60d50956282b37537a788ffa05581dc85cd8fea'
    import verify_source_baseline
    source_baseline = verify_source_baseline.verify()
    assert source_baseline['original_native_abi_verified']
    return {'canonical_version': version, 'mode': 'canonical-default-all-current-graph-fresh-install', 'site': str(site), 'manifest': str(manifest_path), 'manifest_sha256': inventory.digest_file(manifest_path), 'source_digest': frozen['source_digest'], 'compiler_policy_sha256': compiler_policy_sha256, 'compiler_policy_verified': True, 'post_verified': True, 'native_count': len(native), 'visibility': visibility, 'grf512_source_policy_verified': True, 'default_all_verified': True, 'normal_import_verified': False, 'accuracy_verified': False, 'performance_verified': False, 'official_acceptance_complete': False, 'provenance': 'NV v5 real default-all/current install in reused icpx v4 pool; no object copies; NOT independently clean NV'}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--version', default=DEFAULT_VERSION)
    arguments = parser.parse_args()
    if arguments.freeze:
        result = freeze(arguments.version)
        print(json.dumps({'canonical_version': arguments.version, 'ready_source_sha256': result['source_digest'], 'base': result['official_base'], 'files': len(result['files']), 'official_package_pass': False}))
        return
    result = check(arguments.version)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

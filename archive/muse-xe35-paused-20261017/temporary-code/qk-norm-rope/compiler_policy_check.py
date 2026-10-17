import json
from pathlib import Path
import re

REQUIRED_FLAGS = ('-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen')
COMPILERS = {'CMAKE_CXX_COMPILER': ('cxx', 'icpx'), 'CMAKE_C_COMPILER': ('c', 'icx'), 'SYCL_COMPILER': ('sycl', 'icpx'), 'SYCL_HOST_COMPILER': ('host', 'icpx')}


def validate_policy(policy, flags):
    assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command'] is True
    assert set(REQUIRED_FLAGS) <= set(policy['required_flags'])
    for name, (field, executable) in COMPILERS.items():
        path = Path(policy[field])
        assert path.is_absolute() and path.name == executable, f'{name} must use real {executable}'
        assert flags[name]['value'] == str(path), f'Cache/compiler-policy mismatch: {name}'
    assert policy['cxx'] == policy['sycl'] == policy['host']
    assert flags['XPU_SYCL_COMPILER']['value'] == 'icx'


def verify_compiler_policy(canonical, payload, local_path, inventory):
    policy_path = canonical / 'compiler-policy-verified.json'
    assert not policy_path.is_symlink()
    policy = json.loads(policy_path.read_text())
    validate_policy(policy, payload['flags'])
    build = local_path(payload['build'])
    assert canonical.name == 'canonical-v4-icpx'
    expected_pool = canonical.parent.parent / 'canonical-icpx-pool-v4'
    assert build.resolve() == (expected_pool / 'build').resolve(), 'Require independent icpx pool, not old GNU/v2 tree'
    assert local_path(payload['snapshot']).resolve() == (expected_pool / 'source').resolve()
    assert inventory.digest_file(canonical / 'CMakeCache.txt') == payload['cache_sha256']
    assert inventory.digest_file(canonical / 'build.ninja') == payload['ninja_sha256']
    identities = {}
    for language, executable in (('CXX', 'icpx'), ('C', 'icx')):
        candidates = list((build / 'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake'))
        assert len(candidates) == 1
        text = candidates[0].read_text()
        assert f'set(CMAKE_{language}_COMPILER_ID "IntelLLVM")' in text
        compiler = re.search(rf'set\(CMAKE_{language}_COMPILER "([^"]+)"\)', text)
        assert compiler and Path(compiler.group(1)).name == executable
        assert compiler.group(1) == payload['flags'][f'CMAKE_{language}_COMPILER']['value']
        identities[language] = {'path': str(candidates[0]), 'sha256': inventory.digest_file(candidates[0]), 'compiler': compiler.group(1), 'id': 'IntelLLVM'}
    generated = list((build / 'src/CMakeFiles').rglob('*.Release.cmake'))
    assert generated, 'Missing real generated SYCL compiler commands'
    files = [canonical / 'build.ninja', build / 'CMakeFiles/rules.ninja', *generated]
    text = '\n'.join(path.read_text() for path in files)
    for forbidden in ('/usr/bin/c++', '/usr/bin/g++', '/usr/bin/gcc', 'fsycl-host-compiler=gcc'):
        assert forbidden not in text, f'GNU command forbidden: {forbidden}'
    assert policy['cxx'] in text and policy['sycl'] in text and policy['host'] in text
    assert all(flag in text for flag in REQUIRED_FLAGS)
    qk_commands = [path for path in generated if 'FusedQKNormRope' in path.name]
    assert qk_commands, 'Missing actual QK SYCL compiler commands'
    for path in qk_commands:
        command = path.read_text()
        assert policy['sycl'] in command
        assert all(flag in command for flag in REQUIRED_FLAGS), f'QK precision/SYCL flags missing: {path}'
    return {'verified': True, 'compiler_policy_file': str(policy_path), 'compiler_policy_sha256': inventory.digest_file(policy_path), 'policy': policy, 'real_cmake_compiler_identity': identities, 'generated_qk_commands': {str(path): inventory.digest_file(path) for path in qk_commands}, 'archived_cache_sha256': payload['cache_sha256'], 'archived_ninja_sha256': payload['ninja_sha256'], 'build_tree': str(build)}

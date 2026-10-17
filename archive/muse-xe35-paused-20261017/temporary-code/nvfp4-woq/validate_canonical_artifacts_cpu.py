import ast
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
import canonical_check
import verify_source_baseline

parser = argparse.ArgumentParser()
parser.add_argument('--version', default=canonical_check.DEFAULT_VERSION)
parser.add_argument('--previous-version', default='canonical-v3-icpx')
parser.add_argument('--expected-source-delta', default='src/sycl/HCHead.cpp')
args = parser.parse_args()
canonical_check.select_version(args.version)
root = Path(__file__).resolve().parent
python_files = ['canonical_check.py', 'canonical_launch_preflight.py', 'wait_canonical_barrier.py', 'run_canonical_case.py', 'analyze_canonical_results.py', 'validate_canonical_artifacts_cpu.py']
for name in python_files:
    ast.parse((root / name).read_text(), filename=name)
for name in ['launch_canonical_validation.sh', 'run_canonical_validation.sh']:
    subprocess.run(['bash', '-n', str(root / name)], check=True)

cxx = '/opt/gfx-deps/dpcpp/bin/icpx'
fields = {'CMAKE_CXX_COMPILER': cxx, 'CMAKE_C_COMPILER': '/opt/gfx-deps/dpcpp/bin/icx', 'SYCL_COMPILER': cxx, 'SYCL_HOST_COMPILER': cxx, 'XPU_SYCL_COMPILER': 'icx'}
flags = ['-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen']
policy = {'compiler_id': 'IntelLLVM', 'cxx': cxx, 'c': fields['CMAKE_C_COMPILER'], 'sycl': cxx, 'host': cxx, 'torch_SYCL_family': 'icx', 'required_flags': flags, 'no_GNU_command': True}
cache = '\n'.join(f'{name}:STRING={value}' for name, value in fields.items())
command = '  COMMAND = ' + cxx + ' ' + ' '.join(flags) + ' -internal_options -cl-intel-512-GRF-per-thread -o sgl-ops-sycl-blockwise_moe_nvfp4_woq_sycl_device_obj.o'
payload = {'flags': {name: {'value': value} for name, value in fields.items()}, 'cache_sha256': 'synthetic-cache', 'ninja_sha256': 'synthetic-ninja'}


def synthetic_policy_case(record, cache_text=cache, ninja_text=command, accepted=False):
    contents = {'compiler-policy-verified.json': json.dumps(record), 'CMakeCache.txt': cache_text, 'build.ninja': ninja_text}
    hashes = {'compiler-policy-verified.json': 'synthetic-policy', 'CMakeCache.txt': 'synthetic-cache', 'build.ninja': 'synthetic-ninja'}
    with patch.object(Path, 'read_text', lambda path: contents[path.name]), patch.object(canonical_check.inventory, 'digest_file', lambda path: hashes[path.name]):
        try:
            canonical_check.verify_compiler_policy(payload, root / 'synthetic-memory-only')
        except (AssertionError, KeyError):
            assert not accepted
        else:
            assert accepted


synthetic_policy_case(policy, accepted=True)
mutations = [('compiler_id', 'GNU'), ('cxx', '/usr/bin/c++'), ('c', '/usr/bin/cc'), ('sycl', '/opt/gfx-deps/dpcpp/bin/icx'), ('host', '/usr/bin/c++'), ('no_GNU_command', False), ('required_flags', flags[:-1])]
for name, value in mutations:
    invalid = copy.deepcopy(policy)
    invalid[name] = value
    synthetic_policy_case(invalid)
synthetic_policy_case(policy, cache_text=cache.replace(cxx, '/usr/bin/c++'))
synthetic_policy_case(policy, ninja_text=command.replace('512-GRF', '256-GRF'))
synthetic_policy_case(policy, ninja_text=command.replace(cxx, '/opt/gfx-deps/dpcpp/bin/icx'))
synthetic_policy_case(policy, ninja_text=command + ' /usr/bin/g++')

for version in ['canonical-v1', 'canonical-v2', 'canonical-v3', 'canonical-v3-icpx', 'native-diagnostic']:
    try:
        canonical_check.select_version(version)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Old/diagnostic AFTER accepted')

environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
environment.pop('NVFP4_CANONICAL_RELEASE', None)
launcher = ['bash', str(root / 'launch_canonical_validation.sh')]
cases = [
    (launcher + ['--version', 'canonical-v2', 'accuracy', 'v4-cpu-reject-old'], environment),
    (launcher + ['accuracy', 'v4-cpu-reject-no-release'], environment),
    (launcher + ['performance', 'v4-cpu-reject-no-grant'], dict(environment, NVFP4_CANONICAL_RELEASE=str(root / 'nonexistent-release-cpu-only.json'))),
    (launcher + ['accuracy', 'v4-cpu-reject-batch'], dict(environment, BATCH='7', NVFP4_CANONICAL_RELEASE=str(root / 'nonexistent-release-cpu-only.json'))),
]
for arguments, variables in cases:
    completed = subprocess.run(arguments, env=variables, text=True, capture_output=True)
    assert completed.returncode == 2, completed
    print(json.dumps({'CPU_negative_case': arguments[-1], 'actual_exit': completed.returncode, 'stderr': completed.stderr.strip()}))

frozen = canonical_check.freeze(args.version)
old = json.loads((root / f'canonical-source-freeze-{args.previous_version.removeprefix("canonical-")}.json').read_text())
changed = sorted(name for name in set(old['files']) | set(frozen['files']) if old['files'].get(name) != frozen['files'].get(name))
assert changed == [args.expected_source_delta], changed
baseline = verify_source_baseline.verify()
assert baseline['original_native_abi_verified']
print(json.dumps({'canonical_version': args.version, 'source_digest': frozen['source_digest'], 'source_files': len(frozen['files']), 'previous_version': args.previous_version, 'changed_source_files': changed, 'syntax_checked': python_files + ['launch_canonical_validation.sh', 'run_canonical_validation.sh'], 'synthetic_memory_only_policy_cases': 12, 'real_compiler_or_package_acceptance': False, 'source_original_abi_cpu_verified': True, 'full_started': False, 'gpu_started': False, 'waiter_started': False, 'source_freeze_sha256': hashlib.sha256(canonical_check.READY.read_bytes()).hexdigest()}))

import ast
import copy
import json
import os
import subprocess
import sys

from canonical_check import ARTIFACT, CURRENT_VERSION, canonical_directory, verify_reviewed_freeze
from compiler_policy_check import COMPILERS, REQUIRED_FLAGS, validate_policy


def rejected(policy, flags):
    try:
        validate_policy(policy, flags)
    except (AssertionError, KeyError):
        return
    raise AssertionError('Invalid compiler policy was accepted')


reviewed, reviewed_path = verify_reviewed_freeze(CURRENT_VERSION)
assert reviewed['source_digest'] == 'bc9f0d6a7b9ad7c27f205b3cd01f7c2ebac839cdeb7ca8a2d5aa8da65fb4daf2'
policy = {
    'compiler_id': 'IntelLLVM', 'cxx': '/opt/gfx-deps/dpcpp/bin/icpx',
    'c': '/opt/gfx-deps/dpcpp/bin/icx', 'sycl': '/opt/gfx-deps/dpcpp/bin/icpx',
    'host': '/opt/gfx-deps/dpcpp/bin/icpx', 'required_flags': list(REQUIRED_FLAGS),
    'no_GNU_command': True,
}
flags = {name: {'value': policy[field]} for name, (field, executable) in COMPILERS.items()}
flags['XPU_SYCL_COMPILER'] = {'value': 'icx'}
validate_policy(policy, flags)
negative_cases = 0
for field, wrong in (
    ('compiler_id', 'GNU'), ('cxx', '/usr/bin/g++'), ('c', '/usr/bin/gcc'),
    ('sycl', '/opt/gfx-deps/dpcpp/bin/icx'), ('host', '/usr/bin/g++'),
    ('no_GNU_command', False), ('required_flags', ['-fsycl-targets=spir64_gen']),
):
    altered = copy.deepcopy(policy)
    altered[field] = wrong
    rejected(altered, flags)
    negative_cases += 1
for key in (*COMPILERS, 'XPU_SYCL_COMPILER'):
    altered = copy.deepcopy(flags)
    altered[key]['value'] = '/usr/bin/g++'
    rejected(policy, altered)
    negative_cases += 1
old_versions = ('canonical-v1', 'canonical-v2', 'canonical-v3-icpx')
for version in old_versions:
    try:
        canonical_directory(version)
    except AssertionError:
        pass
    else:
        raise AssertionError('Historical AFTER accepted')
for name in (
    'canonical_check.py', 'compiler_policy_check.py', 'wait_gpu_barrier.py',
    'run_suite.py', 'run_preopt.py', 'runtime_check.py', 'audit_legacy_native_compilers.py',
):
    ast.parse((ARTIFACT / name).read_text())
subprocess.run(['bash', '-n', str(ARTIFACT / 'launch_xe35.sh')], check=True)
environment = {**os.environ, 'QK_CANONICAL_VERSION': CURRENT_VERSION, 'PYTHONDONTWRITEBYTECODE': '1'}
assert json.loads((ARTIFACT / 'official-release-policy.json').read_text())['official_gpu_launch_authorized'] is False
paused = subprocess.run(['bash', str(ARTIFACT / 'launch_xe35.sh')], env=environment, capture_output=True, text=True)
assert paused.returncode == 2 and 'launch paused' in paused.stderr
scope = subprocess.run(['bash', str(ARTIFACT / 'launch_xe35.sh')], env={**environment, 'BATCH_SIZE': '9'}, capture_output=True, text=True)
assert scope.returncode == 2 and 'scope error' in scope.stderr and 'launch paused' not in scope.stderr
missing = subprocess.run(
    [sys.executable, str(ARTIFACT / 'canonical_check.py'), '--canonical-version', CURRENT_VERSION],
    env=environment, capture_output=True, text=True,
)
assert missing.returncode != 0, 'Unbuilt QK v4 unexpectedly accepted'
for label, process in (('paused-launch', paused), ('scope-reject', scope), ('missing-evidence', missing)):
    (ARTIFACT / f'v4-icpx-{label}.log').write_text(process.stdout + process.stderr)
    (ARTIFACT / f'v4-icpx-{label}.exit').write_text(f'{process.returncode}\n')
payload = {
    'version': CURRENT_VERSION, 'reviewed_source_digest': reviewed['source_digest'],
    'reviewed_source_freeze': str(reviewed_path),
    'v2_to_v3_only_BuildFlags_delta_verified': True,
    'v3_to_v4_only_HCHead_device_guard_verified': True,
    'main_entire_device_IR_proof_metadata_checked': reviewed['device_ir_proof_sha256'],
    'protected_QK_math_API_reference_unchanged': True,
    'compiler_metadata_unit_positive_cases': 1, 'compiler_metadata_unit_negative_cases': negative_cases,
    'unit_fixtures_are_not_actual_compiler_acceptance': True,
    'real_QK_compiler_policy_file_created_by_worker': False,
    'historical_AFTER_versions_rejected': list(old_versions),
    'paused_launcher_actual_exit': paused.returncode, 'scope_reject_before_GPU_actual_exit': scope.returncode,
    'missing_real_full_package_actual_exit': missing.returncode,
    'SOURCE_build_GPU_waiter_full_launched': False, 'frozen_PR_modified_by_worker': False,
}
(ARTIFACT / 'v4-icpx-preflight-audit.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))

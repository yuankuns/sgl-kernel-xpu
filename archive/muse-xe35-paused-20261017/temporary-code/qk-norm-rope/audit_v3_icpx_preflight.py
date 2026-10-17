import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from canonical_check import ARTIFACT, canonical_directory, verify_reviewed_freeze
from compiler_policy_check import COMPILERS, REQUIRED_FLAGS, validate_policy


def rejected(policy, flags):
    try:
        validate_policy(policy, flags)
    except (AssertionError, KeyError):
        return
    raise AssertionError('Invalid compiler policy was accepted')


reviewed, reviewed_path = verify_reviewed_freeze()
assert reviewed['source_digest'] == 'c5d443b29ff008509726b8183bb553eba7dba1872ac7f067c92bb0e22334b6d6'
policy = {'compiler_id': 'IntelLLVM', 'cxx': '/opt/gfx-deps/dpcpp/bin/icpx', 'c': '/opt/gfx-deps/dpcpp/bin/icx', 'sycl': '/opt/gfx-deps/dpcpp/bin/icpx', 'host': '/opt/gfx-deps/dpcpp/bin/icpx', 'required_flags': list(REQUIRED_FLAGS), 'no_GNU_command': True}
flags = {name: {'value': policy[field]} for name, (field, executable) in COMPILERS.items()}
flags['XPU_SYCL_COMPILER'] = {'value': 'icpx'}
validate_policy(policy, flags)
negative_cases = 0
for field, wrong in (('compiler_id', 'GNU'), ('cxx', '/usr/bin/g++'), ('c', '/usr/bin/gcc'), ('sycl', '/opt/gfx-deps/dpcpp/bin/icx'), ('host', '/usr/bin/g++'), ('no_GNU_command', False), ('required_flags', ['-fsycl-targets=spir64_gen'])):
    altered = copy.deepcopy(policy)
    altered[field] = wrong
    rejected(altered, flags)
    negative_cases += 1
for key in COMPILERS:
    altered = copy.deepcopy(flags)
    altered[key]['value'] = '/usr/bin/g++'
    rejected(policy, altered)
    negative_cases += 1
for version in ('canonical-v1', 'canonical-v2'):
    try:
        canonical_directory(version)
    except AssertionError:
        pass
    else:
        raise AssertionError('Historical AFTER accepted')
for name in ('canonical_check.py', 'compiler_policy_check.py', 'wait_gpu_barrier.py', 'run_suite.py', 'run_preopt.py', 'runtime_check.py'):
    ast.parse((ARTIFACT / name).read_text())
subprocess.run(['bash', '-n', str(ARTIFACT / 'launch_xe35.sh')], check=True)
paused = subprocess.run(['bash', str(ARTIFACT / 'launch_xe35.sh')], env={**os.environ, 'QK_CANONICAL_VERSION': 'canonical-v3-icpx'}, capture_output=True, text=True)
assert paused.returncode == 2 and 'launch paused' in paused.stderr
missing = subprocess.run([sys.executable, str(ARTIFACT / 'canonical_check.py'), '--canonical-version', 'canonical-v3-icpx'], env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
assert missing.returncode != 0, 'Unbuilt QK v3 unexpectedly accepted'
(ARTIFACT / 'v3-icpx-missing-evidence.log').write_text(missing.stdout + missing.stderr)
(ARTIFACT / 'v3-icpx-missing-evidence.exit').write_text(f'{missing.returncode}\n')
payload = {'version': 'canonical-v3-icpx', 'reviewed_source_digest': reviewed['source_digest'], 'reviewed_source_freeze': str(reviewed_path), 'v2_to_v3_only_BuildFlags_delta_verified': True, 'protected_QK_math_API_reference_unchanged': True, 'compiler_metadata_unit_positive_cases': 1, 'compiler_metadata_unit_negative_cases': negative_cases, 'unit_fixtures_are_not_actual_compiler_acceptance': True, 'real_QK_compiler_policy_file_created_by_worker': False, 'historical_AFTER_versions_rejected': ['canonical-v1', 'canonical-v2'], 'paused_launcher_actual_exit': paused.returncode, 'missing_real_full_package_actual_exit': missing.returncode, 'GPU_or_waiter_or_full_launched': False, 'frozen_PR_modified': False}
(ARTIFACT / 'v3-icpx-preflight-audit.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))

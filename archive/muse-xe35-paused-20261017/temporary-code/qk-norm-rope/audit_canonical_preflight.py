import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from canonical_check import ARTIFACT, CANONICAL, READY, TARGET, inventory, local_path, verify_default_command, canonical_directory, verify_reviewed_freeze


def rejected(function, *arguments):
    try:
        function(*arguments)
    except (AssertionError, ValueError):
        return
    raise AssertionError(f'Expected rejection: {arguments}')


frozen, reviewed_path = verify_reviewed_freeze()
assert canonical_directory('canonical-v3-icpx') == ARTIFACT / 'canonical-v3-icpx'
rejected(canonical_directory, 'canonical-v1')
rejected(canonical_directory, 'canonical-v2')
build = ARTIFACT.parent / 'canonical-pool/build'
for prefix in ('/home/syk/', '/root/'):
    value = str(build).replace('/home/syk/', prefix, 1)
    verify_default_command(f'cmake --build {value} --parallel 56', build)
    verify_default_command(f'cmake --build {value} --parallel 8 --config Release', build)
for suffix in ('--target common_ops', '--target all', '--parallel 57', '--parallel 0', '-- -j8', '--config Debug', '; true'):
    rejected(verify_default_command, f'cmake --build {build} {suffix}', build)
rejected(verify_default_command, f'cmake --build {TARGET / "build"} --parallel 8', build)
assert local_path('/root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35') == TARGET
for filename in ('canonical_check.py', 'runtime_check.py', 'run_suite.py', 'wait_gpu_barrier.py'):
    ast.parse((ARTIFACT / filename).read_text())
subprocess.run(['bash', '-n', str(ARTIFACT / 'launch_xe35.sh')], check=True)
scope_results = []
for name, value in [('BATCH_SIZE', '7'), ('BATCHES', '9'), ('ISL', '4095'), ('OSL', '15'), ('TP', '2'), ('EP', '2')]:
    result = subprocess.run(['bash', str(ARTIFACT / 'launch_xe35.sh')], env={**os.environ, name: value}, text=True, capture_output=True)
    assert result.returncode == 2 and 'scope error:' in result.stderr
    scope_results.append({'variable': name, 'value': value, 'exit': result.returncode, 'stderr': result.stderr})
result = subprocess.run(['bash', str(ARTIFACT / 'launch_xe35.sh'), '--tokens', '8'], text=True, capture_output=True)
assert result.returncode == 2 and 'scope error:' in result.stderr
runtime_source = (ARTIFACT / 'runtime_check.py').read_text()
assert 'load_library' not in runtime_source and 'torch.library' not in runtime_source
assert 'runpy.run_module' in runtime_source and 'query_device.default' in runtime_source
assert 'copy' not in (ARTIFACT / 'launch_xe35.sh').read_text()
if not (CANONICAL / 'source-binary-manifest.json').exists():
    result = subprocess.run([sys.executable, str(ARTIFACT / 'canonical_check.py')], text=True, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    (ARTIFACT / f'{CANONICAL.name}-missing-package.log').write_text(result.stdout + result.stderr)
    (ARTIFACT / f'{CANONICAL.name}-missing-package.exit').write_text(f'{result.returncode}\n')
    assert result.returncode != 0 and ('source-binary-manifest.json' in result.stderr or 'source-snapshot.json' in result.stderr)
payload = {'ready_source_sha256': frozen['source_digest'], 'ready_manifest': str(READY), 'default_all_commands_accepted': 4, 'targeted_or_invalid_commands_rejected': 8, 'scope_rejections_before_GPU': scope_results, 'cli_override_rejected': True, 'python_AST_and_bash_syntax_verified': True, 'normal_import_not_fake_registration_or_load_library': True, 'canonical_package_present': (CANONICAL / 'source-binary-manifest.json').exists(), 'official_package_pass': False, 'GPU_work_started': False, 'main_helpers_or_live_scripts_modified': False}
(ARTIFACT / 'canonical-preflight-audit.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))

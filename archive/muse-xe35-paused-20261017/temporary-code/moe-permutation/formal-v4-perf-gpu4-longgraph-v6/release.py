import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
FROZEN = ROOT / 'formal-v4-perf-gpu4-longgraph-v3'
SNAPSHOT = ROOT.parent / 'EXTERNAL_MAINTENANCE_SNAPSHOT_20261017_v1.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_frozen(name, namespace):
    inventory = json.loads((FOLDER / 'frozen-implementation-manifest.json').read_text())
    for filename, expected in inventory['files_sha256'].items():
        if digest(FROZEN / filename) != expected:
            raise RuntimeError('Frozen implementation changed: ' + filename)
    code = (FROZEN / name).read_bytes()
    exec(compile(code, str(FROZEN / name), 'exec'), namespace)


def maintenance_gate():
    fm_proofs = fm_release_gate()
    snapshot = json.loads(SNAPSHOT.read_text())
    if snapshot['external_processes']:
        raise RuntimeError('Coordinator maintenance drain is not empty')
    kernel = subprocess.check_output(['uname', '-r'], text=True).strip()
    loaded = {name: (Path('/sys/module/xe') / name).read_text().strip() for name in ['version', 'srcversion', 'initstate']}
    if kernel != snapshot['running_kernel'] or loaded != snapshot['loaded_modules']['xe']:
        raise RuntimeError('Loaded stack changed since main maintenance snapshot')
    sys.path.insert(0, str(ROOT.parent / 'canonical-tools'))
    from build_pool_icpx_v2 import process_snapshot
    from queue_pool_icpx_v2 import verify_drained
    verify_drained()
    current = process_snapshot()
    names = {'cc1', 'cc1plus', 'gcc', 'g++', 'clang', 'clang++', 'clang-23', 'clang-24', 'icpx', 'icx',
             'ninja', 'make', 'gmake', 'dkms', 'dpkg', 'apt', 'apt-get', 'sycl-post-link', 'llvm-spirv',
             'clang-offload-bu', 'ocloc'}
    active = {str(pid): item for pid, item in current.items() if item['name'] in names}
    zombies = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            name = (directory / 'comm').read_text().strip()
            if fields[0] in ['Z', 'X'] and name in names:
                zombies.append({'pid': int(directory.name), 'name': name, 'state': fields[0], 'starttick': fields[19]})
        except (OSError, ValueError):
            continue
    record = {'epoch': time.time(), 'maintenance_snapshot_sha256': digest(SNAPSHOT), 'running_kernel': kernel,
              'loaded_modules': {'xe': loaded}, 'actual_compiler_processes': active,
              'nonrunning_zombie_processes': zombies, 'canonical_verify_drained_actual_exit': 0, 'FM_release_proofs': fm_proofs}
    if active:
        (FOLDER / 'ACTIVE_MAINTENANCE_BLOCK.json').write_text(json.dumps(record, indent=2) + '\n')
        raise RuntimeError('Actual external/compiler process still active: ' + str(active))
    return record


def fm_release_gate():
    audit_path = ROOT.parent / 'fmha/COORDINATOR_FULL_ICPX_AUDIT_v5.json'
    stage = ROOT.parent / 'fmha/canonical-v5-icpx'
    drain_path = stage / 'COORDINATOR_COMPILER_DRAIN.json'
    quiet_path = ROOT.parent / 'MAINTENANCE_QUIET_AFTER_FM_v3.json'
    audit = json.loads(audit_path.read_text())
    drain = json.loads(drain_path.read_text())
    quiet = json.loads(quiet_path.read_text())
    authorization = json.loads((FOLDER / 'release-authorization.json').read_text())
    proofs = {'FM_audit': digest(audit_path), 'FM_drain': digest(drain_path),
              'quiet3': digest(quiet_path), 'FM_manifest': digest(stage / 'source-binary-manifest.json')}
    if proofs != authorization['FM_release_proofs_sha256']:
        raise RuntimeError('Authorized new FM/quiet evidence changed')
    if audit['component'] != 'fmha' or audit['version'] != 'canonical-v5-icpx' or audit['actual_exit'] != 0 or audit['default_all_actual_exit'] != 0 or audit['install_actual_exit'] != 0:
        raise RuntimeError('FM new full/install/audit is not actual0')
    if audit['installed_and_python_files_rehashed'] != 1694 or audit['native_binary_inventory_rehashed'] != 735 or audit['package_manifest_sha256'] != proofs['FM_manifest']:
        raise RuntimeError('FM independent current install inventory mismatch')
    for name in ['configure.exit', 'default-all-build.exit', 'supervisor.exit']:
        if (stage / name).read_text().strip() != '0':
            raise RuntimeError('FM actual full0 gate mismatch: ' + name)
    post = json.loads((stage / 'source-readonly-postverify.json').read_text())
    if not post['verified'] or 'CANONICAL_INSTALL_ACTUAL_EXIT=0' not in (stage / 'install.log').read_text():
        raise RuntimeError('FM current install/source-post0 required')
    if drain['actual_exit'] != 0 or drain['full_build_and_independent_audit_exit'] != 0 or drain['active_compile_or_maintenance']:
        raise RuntimeError('FM compiler-drain proof is not actual0/empty')
    if quiet['actual_exit'] != 0 or quiet['continuous_quiet_seconds'] < 60:
        raise RuntimeError('Main maintenance quiet3 bounded60 gate missing')
    snapshot = json.loads(SNAPSHOT.read_text())
    expected_stack = {'running_kernel': snapshot['running_kernel'], 'loaded_modules': snapshot['loaded_modules']}
    if quiet['loaded_stack'] != expected_stack or any(sample['active_processes'] or sample['loaded_stack'] != expected_stack for sample in quiet['samples']):
        raise RuntimeError('Quiet3 active processes or stack changed')
    if authorization['epoch'] <= quiet['samples'][-1]['epoch']:
        raise RuntimeError('New release must follow quiet3 actual completion')
    return proofs

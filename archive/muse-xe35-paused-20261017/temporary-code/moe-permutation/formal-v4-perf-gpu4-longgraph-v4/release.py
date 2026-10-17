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
              'nonrunning_zombie_processes': zombies, 'canonical_verify_drained_actual_exit': 0}
    if active:
        (FOLDER / 'ACTIVE_MAINTENANCE_BLOCK.json').write_text(json.dumps(record, indent=2) + '\n')
        raise RuntimeError('Actual external/compiler process still active: ' + str(active))
    return record

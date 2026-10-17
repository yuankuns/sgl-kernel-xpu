import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

import build_pool_icpx_v2 as pool
from overlay_snapshot import manifest_digest, source_inventory


VERSION = 'canonical-v5-icpx'
COMPONENT = 'nvfp4-woq'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def host_build():
    artifact = pool.ROOT / COMPONENT
    evidence = artifact / VERSION
    evidence.mkdir(exist_ok=False)
    source = pool.ROOT.parent / 'pr/muse-nvfp4-woq-xe35'
    candidate_path = artifact / 'v5-typing-import-preparation/candidate-source-freeze-v5.json'
    candidate = json.loads(candidate_path.read_text())
    old = json.loads((artifact / 'canonical-source-freeze-v4-icpx.json').read_text())
    files, selection = source_inventory(source)
    assert files == candidate['files'] and selection == candidate['selection']
    assert manifest_digest(files) == candidate['source_digest']
    assert files.keys() == old['files'].keys() and selection == old['selection']
    changed = [name for name in files if files[name] != old['files'][name]]
    assert changed == ['python/sgl_kernel/nvfp4_moe.py'], changed
    previous_python = artifact / 'v5-typing-import-preparation/before-nvfp4_moe.py'
    current_python = source / changed[0]
    assert previous_python.read_text().replace('from typing import Tuple\n', 'from typing import Optional, Tuple\n', 1) == current_python.read_text()
    static = json.loads((artifact / 'v5-typing-import-preparation/annotation-static-CPU-check.json').read_text())
    assert static['actual_static_check_exit'] == 0
    assert static['function_signatures_defaults_decorators_bodies_unchanged'] is True
    assert static['after_sha256'] == digest(current_python)
    with pool.GLOBAL_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = pool.process_snapshot()
        live_compilers = {pid: item for pid, item in processes.items()
                          if item['name'] in {'icpx', 'icx', 'ninja', 'clang', 'clang++'}
                          and str(pool.ROOT) in item['command']}
        assert not live_compilers, live_compilers
        assert pool.memory_available_gib() >= pool.JOBS * 8 + pool.MIN_AVAILABLE_GIB
        freeze_path = artifact / 'canonical-source-freeze-v5-icpx.json'
        with freeze_path.open('x') as output:
            output.write(json.dumps(candidate, indent=2) + '\n')
        pool.save_json(evidence / 'preflight.json', {
            'epoch': time.time(), 'issuer': 'main-coordinator', 'source_delta': changed,
            'native_source_unchanged': True, 'frozen_v4_stage_untouched': True,
            'candidate_sha256': digest(candidate_path), 'freeze_sha256': digest(freeze_path),
            'builder_dependency_sha256': digest(pool.TOOLS / 'build_pool_icpx_v2.py'),
            'available_gib': pool.memory_available_gib(), 'live_compilers': live_compilers,
            'outer_jobs': pool.JOBS, 'inner_jobs': 1, 'cpu_limit': 128, 'GPU_work': False,
        })
        previous_pointer = pool.POOL / 'latest-snapshot-manifest.json'
        command = [sys.executable, str(pool.TOOLS / 'overlay_snapshot.py'),
                   '--source', str(source), '--destination', str(pool.SNAPSHOT),
                   '--manifest', str(evidence / 'source-snapshot.json')]
        if previous_pointer.exists():
            command.extend(['--previous-manifest', json.loads(previous_pointer.read_text())['path']])
        subprocess.run(command, check=True)
        pool.save_json(previous_pointer, {'path': str(evidence / 'source-snapshot.json')})
        script = str(pool.TOOLS / 'build_nv_typing_icpx_v5.py').replace('/home/syk/', '/root/', 1)
        wrapper = [pool.WRAPPER, '--workdir', '/root/sgl-kernel-xpu-up', '--',
                   'taskset', '-c', '0-127', '/opt/gfx-deps/venv/bin/python', script, '--container-build']
        environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015',
                           PYTHONDONTWRITEBYTECODE='1')
        pool.run_command(wrapper, evidence / 'supervisor', environment)
        subprocess.run([*command, '--verify-only'], check=True)
        pool.save_json(evidence / 'source-readonly-postverify.json', {
            'verified': True, 'epoch': time.time(), 'source': str(source),
            'snapshot_manifest': str(evidence / 'source-snapshot.json'),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--container-build', action='store_true')
    args = parser.parse_args()
    if args.container_build:
        pool.container_build(COMPONENT, VERSION)
    else:
        host_build()


if __name__ == '__main__':
    main()

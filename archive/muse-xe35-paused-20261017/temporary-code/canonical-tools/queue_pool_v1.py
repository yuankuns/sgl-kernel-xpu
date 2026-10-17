import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from overlay_snapshot import manifest_digest, source_inventory


ROOT = Path(__file__).resolve().parent.parent


def live(pid, starttime):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and fields[19] == starttime


def verify_drained():
    offenders = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            name = (directory / 'comm').read_text().strip()
        except OSError:
            continue
        if fields[0] != 'Z' and name in {'ninja', 'cmake', 'icx', 'icpx', 'clang++', 'clang-23', 'clang-24', 'g++', 'cc1plus', 'ocloc'} and '/canonical-pool/' in command:
            offenders.append({'pid': int(directory.name), 'name': name, 'command': command})
    if offenders:
        raise RuntimeError(f'Canonical process tree still active; do not overlay: {offenders}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--components', nargs='+', required=True)
    args = parser.parse_args()
    fields = Path(f'/proc/{args.wait_pid}/stat').read_text().rsplit(')', 1)[1].split()
    command = Path(f'/proc/{args.wait_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if 'build_pool_v1.py --component bf16-gemm' not in command:
        raise RuntimeError('Expected active BF16 canonical host supervisor identity')
    starttime = fields[19]
    queue_dir = ROOT / 'canonical-pool/queue-v1'
    queue_dir.mkdir(exist_ok=False)
    with (ROOT / 'canonical-pool/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (queue_dir / 'plan.json').write_text(json.dumps({'wait_pid': args.wait_pid, 'wait_starttime': starttime, 'components': args.components, 'gpu_work': False, 'automatic_restart_on_failure': False, 'maximum_outer_jobs': 56, 'inner_jobs': 1}, indent=2) + '\n')
        while live(args.wait_pid, starttime):
            time.sleep(20)
        bf16 = ROOT / 'bf16-gemm/canonical-v1'
        if (bf16 / 'supervisor.exit').read_text().strip() != '0' or not (bf16 / 'source-readonly-postverify.json').is_file():
            raise RuntimeError('Original BF16 canonical full/install/postverify did not succeed; queue stopped without restart')
        for component in args.components:
            if component not in {'fmha', 'qk-norm-rope', 'residual-norm', 'routing-quant', 'moe-permutation', 'memory', 'position', 'logits-vocab'}:
                raise RuntimeError('Module not authorized in this frozen queue')
            verify_drained()
            artifact = ROOT / component
            frozen = json.loads((artifact / 'canonical-source-freeze-v1.json').read_text())
            files, selection = source_inventory(Path(frozen['source']))
            if files != frozen['files'] or selection != frozen['selection'] or manifest_digest(files) != frozen['source_digest']:
                raise RuntimeError(f'Frozen source changed: {component}; do not build another payload')
            command = [sys.executable, str(ROOT / 'canonical-tools/build_pool_v1.py'), '--component', component]
            with (queue_dir / f'{component}.log').open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            (queue_dir / f'{component}.exit').write_text(str(result.returncode) + '\n')
            if result.returncode:
                raise RuntimeError(f'{component} actual build supervisor exited {result.returncode}; no retry/no successor')
            if not (artifact / 'canonical-v1/source-readonly-postverify.json').is_file():
                raise RuntimeError('Source-readonly postverify missing')
            verify_drained()
        (queue_dir / 'complete.json').write_text(json.dumps({'all_queued_default_all_and_fresh_install_completed': True, 'gpu_acceptance_complete': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()

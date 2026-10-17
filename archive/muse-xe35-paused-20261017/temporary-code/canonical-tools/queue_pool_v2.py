import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from overlay_snapshot import manifest_digest, source_inventory
from queue_pool_v1 import ROOT, live, verify_drained


COMPONENTS = ('fmha', 'qk-norm-rope', 'mxfp4-w4a4', 'residual-norm', 'nvfp4-woq', 'moe-permutation', 'routing-quant', 'memory', 'position', 'logits-vocab')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    fields = Path(f'/proc/{args.wait_pid}/stat').read_text().rsplit(')', 1)[1].split()
    command = Path(f'/proc/{args.wait_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if 'build_pool_v1.py --component bf16-gemm' not in command:
        raise RuntimeError('Expected active BF16 canonical host supervisor identity')
    starttime = fields[19]
    queue_dir = ROOT / 'canonical-pool/queue-v2'
    queue_dir.mkdir(exist_ok=False)
    with (ROOT / 'canonical-pool/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {'wait_pid': args.wait_pid, 'wait_starttime': starttime, 'components': list(COMPONENTS), 'gpu_work': False, 'automatic_restart_on_failure': False, 'maximum_outer_jobs': 56, 'inner_jobs': 1, 'superseded_queue_v1_actual_exit': 143, 'bf16_build_interrupted': False}
        (queue_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        while live(args.wait_pid, starttime):
            time.sleep(20)
        bf16 = ROOT / 'bf16-gemm/canonical-v1'
        if (bf16 / 'supervisor.exit').read_text().strip() != '0' or not (bf16 / 'source-readonly-postverify.json').is_file():
            raise RuntimeError('BF16 canonical full/install/postverify did not succeed; no restart')
        for component in COMPONENTS:
            verify_drained()
            artifact = ROOT / component
            frozen = json.loads((artifact / 'canonical-source-freeze-v1.json').read_text())
            files, selection = source_inventory(Path(frozen['source']))
            if files != frozen['files'] or selection != frozen['selection'] or manifest_digest(files) != frozen['source_digest']:
                raise RuntimeError(f'Frozen source changed: {component}; do not silently build another payload')
            command = [sys.executable, str(ROOT / 'canonical-tools/build_pool_v1.py'), '--component', component]
            with (queue_dir / f'{component}.log').open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            (queue_dir / f'{component}.exit').write_text(str(result.returncode) + '\n')
            if result.returncode:
                raise RuntimeError(f'{component} actual build exit {result.returncode}; no retry or successor')
            if not (artifact / 'canonical-v1/source-readonly-postverify.json').is_file():
                raise RuntimeError('Source-readonly postverify missing')
            verify_drained()
        (queue_dir / 'complete.json').write_text(json.dumps({'all_queued_default_all_and_fresh_install_completed': True, 'gpu_acceptance_complete': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()

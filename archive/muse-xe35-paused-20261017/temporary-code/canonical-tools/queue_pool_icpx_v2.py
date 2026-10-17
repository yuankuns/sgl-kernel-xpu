import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_pool_icpx_v2 import POOL, ROOT, VERSION, process_snapshot
from overlay_snapshot import manifest_digest, source_inventory


COMPONENTS = ('bf16-gemm', 'fmha', 'qk-norm-rope', 'mxfp4-w4a4', 'residual-norm', 'nvfp4-woq', 'moe-permutation', 'routing-quant', 'memory', 'position', 'logits-vocab')


def verify_drained():
    offenders = []
    for process_id, item in process_snapshot().items():
        if item['name'] not in {'ninja', 'cmake', 'icx', 'icpx', 'clang++', 'clang-23', 'clang-24', 'g++', 'cc1plus', 'ocloc'}:
            continue
        if any(marker in item['command'] for marker in ('/canonical-pool/', '/canonical-icpx-pool/', '/canonical-icpx-pool-v4/')):
            offenders.append({'pid': process_id, **item})
    if offenders:
        raise RuntimeError(f'Canonical build descendants still live; no overlay: {offenders}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    arguments = parser.parse_args()
    verify_drained()
    for component in COMPONENTS:
        frozen = json.loads((ROOT / component / 'canonical-source-freeze-v4-icpx.json').read_text())
        files, selection = source_inventory(Path(frozen['source']))
        if files != frozen['files'] or selection != frozen['selection'] or manifest_digest(files) != frozen['source_digest']:
            raise RuntimeError(f'Reviewed icpx source changed: {component}')
    if arguments.prepare_only:
        print(json.dumps({'reviewed_freezes': list(COMPONENTS), 'gpu_work': False, 'full_build_started': False}))
        return
    POOL.mkdir(exist_ok=True)
    queue = POOL / 'queue-v1'
    queue.mkdir(exist_ok=False)
    with (ROOT / 'canonical-pool/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {'epoch': time.time(), 'version': VERSION, 'components': list(COMPONENTS), 'cpu_affinity': list(range(128)), 'outer_jobs': 56, 'inner_jobs': 1, 'gpu_work': False, 'automatic_restart_on_failure': False, 'mixed_compiler_queue_actual_exit': 143, 'first_icpx_v3_queue_actual_exit': 1, 'reviewed_manual_restart_after_device_IR_equivalence_proof': True, 'first_family_clean_subsequent_incremental': True}
        (queue / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        for component in COMPONENTS:
            verify_drained()
            command = [sys.executable, str(ROOT / 'canonical-tools/build_pool_icpx_v2.py'), '--component', component]
            (queue / f'{component}.command.json').write_text(json.dumps(command) + '\n')
            with (queue / f'{component}.log').open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            (queue / f'{component}.exit').write_text(str(result.returncode) + '\n')
            if result.returncode:
                raise RuntimeError(f'{component} actual supervisor exit {result.returncode}; no retry/no successor')
            evidence = ROOT / component / VERSION
            postverify = json.loads((evidence / 'source-readonly-postverify.json').read_text())
            policy = json.loads((evidence / 'compiler-policy-verified.json').read_text())
            if postverify.get('verified') is not True or policy.get('compiler_id') != 'IntelLLVM' or policy.get('no_GNU_command') is not True:
                raise RuntimeError('Missing verified source or genuine icpx compiler policy')
            verify_drained()
        (queue / 'complete.json').write_text(json.dumps({'all_default_graph_builds_and_fresh_installs_complete': True, 'gpu_acceptance_complete': False}) + '\n')


if __name__ == '__main__':
    main()

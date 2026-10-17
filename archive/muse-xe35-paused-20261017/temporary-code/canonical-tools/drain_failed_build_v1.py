import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path

from build_pool_v1 import BUILD, POOL, ROOT, SNAPSHOT, descendants, process_snapshot
from overlay_snapshot import verify_inventory


COMPONENTS = ('bf16-gemm', 'fmha', 'qk-norm-rope', 'mxfp4-w4a4', 'residual-norm', 'nvfp4-woq', 'moe-permutation', 'routing-quant', 'memory', 'position', 'logits-vocab')
COMPILER_NAMES = {'ninja', 'cmake', 'icx', 'icpx', 'clang++', 'clang-23', 'clang-24', 'g++', 'cc1plus', 'ocloc', 'python', 'python3', 'python3.12'}


def still_live(captured):
    current = process_snapshot()
    return {process_id: item for process_id, item in captured.items() if process_id in current and current[process_id]['starttime'] == item['starttime']}


def main():
    parser = argparse.ArgumentParser(description='Coordinator-only cleanup of a terminal failed canonical build; no build, overlay or GPU work')
    parser.add_argument('--component', choices=COMPONENTS, required=True)
    parser.add_argument('--version', default='canonical-v1')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if '/' in args.version or args.version in {'.', '..'}:
        raise SystemExit('Invalid evidence version')
    evidence = ROOT / args.component / args.version
    status = int((evidence / 'default-all-build.exit').read_text().strip())
    if status == 0:
        raise SystemExit('Refusing cleanup of a successful build')
    failed_manifest_path = evidence / 'source-snapshot.json'
    failed_manifest = json.loads(failed_manifest_path.read_text())
    if Path(failed_manifest['source']).name != f'muse-{args.component}-xe35':
        raise SystemExit('Wrong failed module identity')
    pointer = json.loads((POOL / 'latest-snapshot-manifest.json').read_text())
    expected_pointer = str(failed_manifest_path).replace('/root/', '/home/syk/', 1)
    if pointer['path'] not in {str(failed_manifest_path), expected_pointer}:
        raise SystemExit('Canonical source has moved to another module; do not signal its processes')
    with (POOL / 'full-pool.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inventory(SNAPSHOT, failed_manifest['files'])
        current = process_snapshot()
        roots = {
            process_id for process_id, item in current.items()
            if item['name'] in COMPILER_NAMES
            and (str(BUILD) in item['command'] or str(SNAPSHOT) in item['command'])
        }
        selected = set()
        for process_id in roots:
            selected.update(descendants(process_id, current))
        captured = {process_id: current[process_id] for process_id in selected}
        audit = {'epoch': time.time(), 'component': args.component, 'failed_build_actual_exit': status, 'dry_run': args.dry_run, 'snapshot_source_digest': failed_manifest['source_digest'], 'captured': captured, 'signals': [], 'no_automatic_retry': True, 'gpu_work': False}
        if not args.dry_run:
            for process_id, item in still_live(captured).items():
                if item['name'] == 'ninja':
                    os.kill(process_id, signal.SIGINT)
                    audit['signals'].append({'pid': process_id, 'signal': 'SIGINT'})
            time.sleep(5)
            for signal_name in ('SIGTERM', 'SIGKILL'):
                for process_id in still_live(captured):
                    try:
                        os.kill(process_id, getattr(signal, signal_name))
                        audit['signals'].append({'pid': process_id, 'signal': signal_name})
                    except ProcessLookupError:
                        pass
                time.sleep(3)
        audit['remaining_live'] = still_live(captured)
        path = evidence / f'failed-build-drain-{time.time_ns()}.json'
        path.write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({'audit': str(path), 'captured': len(captured), 'remaining_live': len(audit['remaining_live']), 'dry_run': args.dry_run}))
        if not args.dry_run and audit['remaining_live']:
            raise SystemExit('Failed canonical subtree did not drain; never overlay or restart')


if __name__ == '__main__':
    main()

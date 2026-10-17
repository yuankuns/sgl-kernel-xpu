import argparse
import json
import math
import os
import pathlib
import subprocess
import time

import canonical_check

parser = argparse.ArgumentParser()
parser.add_argument('--version', default=canonical_check.DEFAULT_VERSION)
parser.add_argument('--phase', choices=('accuracy', 'performance'), required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--barrier', type=pathlib.Path, required=True)
parser.add_argument('--barrier-id', required=True)
parser.add_argument('--grant', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
canonical_check.select_version(args.version)
for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
    if name in os.environ and os.environ[name] != expected:
        raise RuntimeError(f'{name} must be {expected} before GPU work')
if not args.run or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.run):
    raise RuntimeError('Unsafe run name')
if args.barrier.exists():
    raise RuntimeError('Fresh barrier required; do not reuse an already published file')
if args.phase == 'performance' and not args.grant:
    raise RuntimeError('GPU5 requires explicit main grant after FMHA drain before waiter start')
if args.phase == 'accuracy' and args.grant:
    raise RuntimeError('GPU9 precision does not accept a performance grant')
result = canonical_check.check(args.version)
gpu = 9 if args.phase == 'accuracy' else 5
if args.grant:
    grant = json.loads(args.grant.read_text())
    expected = {'issuer': 'main', 'component': 'nvfp4-woq', 'canonical_version': args.version, 'source_digest': result['source_digest'], 'gpu': 5, 'phase': 'performance', 'fmha_complete': True, 'canonical_manifest_sha256': result['manifest_sha256'], 'compiler_policy_sha256': result['compiler_policy_sha256']}
    assert all(grant.get(name) == value for name, value in expected.items()), 'Invalid main GPU5 grant'
ready = {'pid': os.getpid(), 'canonical_version': args.version, 'phase': args.phase, 'run': args.run, 'gpu': gpu, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'barrier': str(args.barrier), 'barrier_id': args.barrier_id, 'source_digest': result['source_digest'], 'canonical_manifest_sha256': result['manifest_sha256'], 'default_all_actual_exit': 0, 'current_install_actual_exit': 0, 'compiler_policy_sha256': result['compiler_policy_sha256'], 'post_verified': True, 'ready': True, 'gpu_lock_held': False, 'status': 'CPU-only waiter; main fresh epoch required; no GPU initialized'}
ready_path = root / f'{args.run}-barrier-ready.json'
if ready_path.exists():
    raise RuntimeError('Use a new waiter run; preserve original ready/raw status')
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
print(json.dumps(ready), flush=True)
while not args.barrier.exists():
    time.sleep(0.5)
release = json.loads(args.barrier.read_text())
expected = {**canonical_check.release_fields(result, args.phase, args.run), 'barrier_id': args.barrier_id}
assert all(release.get(name) == value for name, value in expected.items()), 'Invalid fresh canonical release fields'
if args.phase == 'performance':
    assert release.get('fmha_complete') is True
deadline = float(release['start_epoch'])
if not math.isfinite(deadline) or deadline <= time.time():
    raise RuntimeError('Fresh shared epoch must be finite and in the future')
ready.update({'start_epoch': deadline, 'status': 'fresh epoch received; still CPU-only'})
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
time.sleep(max(0, deadline - time.time()))
command = ['bash', str(root / 'launch_canonical_validation.sh'), '--version', args.version, args.phase, args.run]
if args.grant:
    command.append(str(args.grant))
ready.update({'actual_launcher_start_epoch': time.time(), 'status': 'canonical launcher started at main epoch'})
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
environment = dict(os.environ, NVFP4_CANONICAL_RELEASE=str(args.barrier.resolve()))
status = subprocess.call(command, env=environment)
ready.update({'actual_exit': status, 'ready': False, 'status': 'terminal; actual wrapper released, logical lease remains main-controlled'})
ready_path.write_text(json.dumps(ready, indent=2) + '\n')
raise SystemExit(status)

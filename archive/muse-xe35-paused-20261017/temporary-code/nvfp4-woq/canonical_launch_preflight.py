import argparse
import json
import os
import pathlib
import math
import time

import canonical_check

parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, choices=('accuracy', 'performance'))
parser.add_argument('--run', required=True)
parser.add_argument('--grant', type=pathlib.Path)
parser.add_argument('--release', type=pathlib.Path, required=True)
parser.add_argument('--version', default=canonical_check.DEFAULT_VERSION)
args = parser.parse_args()
canonical_check.select_version(args.version)
for name, expected in {'BATCHES': '8', 'BATCH': '8', 'BATCH_SIZE': '8', 'SERVING_BATCH': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
    if name in os.environ and os.environ[name] != expected:
        raise RuntimeError(f'{name} must be {expected} before GPU work')
if args.phase == 'performance' and not args.grant:
    raise RuntimeError('No GPU5 performance without explicit main grant after FMHA')
if not args.run or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.run):
    raise RuntimeError('Unsafe run name')
root = pathlib.Path(__file__).resolve().parent
canonical_check.select_version(args.version)
if (root / f'{args.run}-canonical.log').exists() or (root / args.version / 'results' / f'{args.run}-production').exists():
    raise RuntimeError('Use a new run name; raw evidence cannot be overwritten')
result = canonical_check.check(args.version)
release = json.loads(args.release.read_text())
expected_release = canonical_check.release_fields(result, args.phase, args.run)
if any(release.get(name) != value for name, value in expected_release.items()):
    raise RuntimeError('Fresh main versioned release does not match actual NV package/compiler/source')
if not release.get('barrier_id'):
    raise RuntimeError('Fresh main barrier ID required')
epoch = float(release['start_epoch'])
if not math.isfinite(epoch) or epoch > time.time():
    raise RuntimeError('Launcher must wait for the main shared epoch')
if args.phase == 'performance':
    grant = json.loads(args.grant.read_text())
    expected = {'issuer': 'main', 'component': 'nvfp4-woq', 'canonical_version': args.version, 'source_digest': result['source_digest'], 'gpu': 5, 'phase': 'performance', 'fmha_complete': True, 'canonical_manifest_sha256': result['manifest_sha256'], 'compiler_policy_sha256': result['compiler_policy_sha256']}
    for name, value in expected.items():
        if grant.get(name) != value:
            raise RuntimeError(f'Invalid main GPU5 grant: {name}')
else:
    if args.grant:
        raise RuntimeError('GPU9 accuracy does not use a performance grant')
record = {**expected_release, 'compiler_policy_verified': True, 'default_all_verified': True, 'gpu_started': False, 'grant': str(args.grant) if args.grant else None, 'release': str(args.release), 'barrier_id': release['barrier_id'], 'start_epoch': epoch}
(root / f'{args.run}-canonical-preflight.json').write_text(json.dumps(record, indent=2) + '\n')
print(result['manifest_sha256'])

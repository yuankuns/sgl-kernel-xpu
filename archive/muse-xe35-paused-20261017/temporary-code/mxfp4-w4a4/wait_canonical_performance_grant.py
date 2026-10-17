import argparse
import json
import math
import pathlib
import re
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=('mxfp4-w4a4', 'routing-quant'), required=True)
    parser.add_argument('--version', default='canonical-v4-icpx')
    parser.add_argument('--phase', choices=('precision', 'performance'), default='performance')
    args = parser.parse_args()
    if not re.fullmatch(r'canonical-v[1-9][0-9]*-icpx', args.version):
        parser.error('New runtime grants require canonical-vN-icpx')
    if args.version == 'canonical-v3-icpx':
        parser.error('Failed canonical-v3-icpx is retired; no runtime grant accepted')
    gpu = 8 if args.phase == 'precision' else (6 if args.module == 'mxfp4-w4a4' else 7)
    artifact = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017') / args.module
    evidence = artifact / args.version
    if (evidence / 'normal-package-check.exit').read_text().strip() != '0':
        raise RuntimeError('Normal-package checker has not genuinely succeeded')
    checked = json.loads((evidence / 'normal-package-check.json').read_text())
    requirements = [('module', args.module), ('version', args.version), ('default_all_and_fresh_current_install_verified', True), ('normal_package_import_verified', True), ('xpu_initialized', False)]
    for name, expected in requirements:
        if checked.get(name) != expected:
            raise RuntimeError(f'Invalid normal-package checker evidence: {name}')
    if checked.get('compiler_policy', {}).get('compiler_id') != 'IntelLLVM':
        raise RuntimeError('No verified real IntelLLVM normal package')
    freeze = json.loads((artifact / ('canonical-source-freeze-' + args.version.removeprefix('canonical-') + '.json')).read_text())
    if checked['source_digest'] != freeze['source_digest']:
        raise RuntimeError('Normal-package checker did not validate this version complete freeze')
    if (evidence / 'source-baseline-check.exit').read_text().strip() != '0':
        raise RuntimeError('Real original source baseline checker has not succeeded')
    baseline = json.loads((evidence / 'source-baseline-check.json').read_text())
    if baseline.get('module') != args.module or baseline.get('version') != args.version or baseline.get('xpu_initialized') is not False:
        raise RuntimeError('Baseline checker evidence mismatch')
    version_tag = args.version.removeprefix('canonical-')
    barrier = artifact / f'gpu{gpu}-canonical-{args.phase}-barrier-{version_tag}.json'
    ready = {'module': args.module, 'version': args.version, 'gpu': gpu, 'phase': 'official-' + args.phase, 'barrier': str(barrier), 'scope': 'B8/ISL4096/OSL16/TP1/EP1', 'source_digest': checked['source_digest'], 'manifest_sha256': checked['manifest_sha256'], 'status': 'Waiting for explicit main lease grant; no GPU lock held', 'predecessor': 'QK' if gpu == 6 else ('Residual' if gpu == 7 else 'Previous GPU8 precision lease')}
    (artifact / args.version / (args.phase + '-ready.json')).write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready), flush=True)
    while not barrier.exists():
        time.sleep(1)
    record = json.loads(barrier.read_text())
    requirements = [('gpu', gpu), ('module', args.module), ('version', args.version), ('coordinator_release', True), ('performance_authorized', args.phase == 'performance'), ('source_digest', ready['source_digest']), ('scope', ready['scope'])]
    if args.phase == 'precision':
        requirements.append(('precision_authorized', True))
    for name, expected in requirements:
        if record.get(name) != expected:
            raise RuntimeError(f'Invalid explicit main grant: {name}')
    epoch = float(record['start_epoch'])
    if not math.isfinite(epoch) or epoch <= 0:
        raise RuntimeError('Invalid main grant epoch')
    time.sleep(max(0, epoch - time.time()))
    print(f'Explicit main canonical normal-package {args.phase} GPU{gpu} grant accepted', flush=True)


if __name__ == '__main__':
    main()

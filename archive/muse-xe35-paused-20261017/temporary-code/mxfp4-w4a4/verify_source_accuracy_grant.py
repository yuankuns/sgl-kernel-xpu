import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare')
    action.add_argument('--seal')
    options = parser.parse_args()
    evidence = Path(options.prepare or options.seal)
    artifact = Path(__file__).resolve().parent
    stage = artifact / 'fresh-source-icpx-native-v2'
    grant = evidence / 'gpu8-source-precision-grant-v1.json'
    required = [stage / 'cpu-native-acceptance.json', stage / 'source-native-package-manifest.json', stage / 'source-preverify.json',
                stage / 'source-postverify.json', artifact / 'source-package-provenance.json', artifact / 'source-before-provenance-v3-icpx.json']
    required += [artifact / name for name in ['source_accuracy_only.py', 'compare_source_accuracy.py', 'run_source_accuracy_only_v1.sh', 'launch_source_accuracy_only_v1.sh', 'check_source_precision_package.py']]
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014')
    required += [source / 'tests/test_mxfp4_moe_w4a4.py', source / 'benchmark/accuracy_reference.py']
    hashes = {str(path): digest(path) for path in required}
    if options.prepare:
        assert int((stage / 'cpu-native-acceptance.exit').read_text()) == 0
        payload = {'authorization': 'latest explicit user grant: GPU8 historical genuine SOURCE versus fresh icpx SOURCE precision only; not canonical/PORT',
                   'release': True, 'grant_epoch': time.time(), 'gpu': 8, 'CPU_affinity': '64-79', 'scope': 'B8/ISL4096/OSL16/TP1/EP1',
                   'source_revision': '54216962c83b2d489702fd38c58d71116f3a6969', 'production_shapes': 5,
                   'manifest_and_scripts_sha256': hashes, 'performance_authorized': False, 'AFTER_comparison_authorized': False,
                   'routing_GPU_successor_authorized': False, 'sealed_package_checks': False}
        grant.write_text(json.dumps(payload, indent=2) + '\n')
    else:
        payload = json.loads(grant.read_text())
        assert payload['manifest_and_scripts_sha256'] == hashes and payload['release'] and payload['gpu'] == 8
        packages = {}
        for kind in ['historical', 'fresh']:
            assert int((evidence / f'package-{kind}.exit').read_text()) == 0
            check = evidence / f'package-{kind}.json'
            metadata = json.loads(check.read_text())
            assert not metadata['GPU_initialized']
            packages[kind] = {'check_sha256': digest(check), 'site': metadata['site'], 'mapped_native_sha256': metadata['mapped_native_sha256'],
                              'package_files_sha256': metadata['package_files_sha256']}
        payload['sealed_package_checks'] = packages
        grant.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'grant': str(grant), 'grant_epoch': payload['grant_epoch'], 'sealed': bool(payload['sealed_package_checks'])}))


if __name__ == '__main__':
    main()

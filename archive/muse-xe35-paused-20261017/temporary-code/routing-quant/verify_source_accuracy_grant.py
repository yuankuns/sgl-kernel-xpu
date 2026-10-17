import argparse
import fcntl
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
    parser.add_argument('--prepare', action='store_true')
    options = parser.parse_args()
    artifact = Path(__file__).resolve().parent
    evidence = artifact / 'historical-fresh-source-precision-v1'
    previous = artifact.parent / 'mxfp4-w4a4/historical-fresh-source-precision-v3'
    for name in ['gpu-runtime', 'host-launch', 'source-paired-precision', 'original-40-historical', 'original-40-fresh']:
        assert int((previous / (name + '.exit')).read_text()) == 0, name
    drain_path = previous / 'gpu8-actual-drain.json'
    drain = json.loads(drain_path.read_text())
    assert drain['GPU8_released'] and drain['actual_handle_exit'] == 0
    assert all(not Path('/proc', str(pid)).exists() for pid in drain['previous_owner_pids_absent'])
    with open('/tmp/intel-gpu-8.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)
    stage = artifact / 'fresh-source-icpx-native-v2'
    for name in ['configure', 'native-build', 'site-package-install', 'site-package-import', 'isolated-site-stage', 'container', 'wrapper', 'supervisor', 'cpu-native-acceptance']:
        assert int((stage / (name + '.exit')).read_text()) == 0, name
    assert all(json.loads((stage / 'source-postverify.json').read_text()).values())
    assert json.loads((stage / 'native-build.resource-summary.json').read_text())['drained']
    required = [stage / 'cpu-native-acceptance.json', stage / 'source-native-package-manifest.json', stage / 'source-preverify.json',
                stage / 'source-postverify.json', artifact / 'source-package-provenance.json', artifact / 'source-before-provenance-v3-icpx.json', drain_path]
    required += [artifact / name for name in ['source_accuracy_only.py', 'compare_source_accuracy.py', 'run_source_accuracy_only_v1.sh', 'launch_source_accuracy_only_v1.sh', 'check_source_precision_package.py']]
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    required += [source / 'tests' / name for name in ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4.py', 'test_topk_sigmoid.py', 'test_routing_benchmark_scope.py']]
    hashes = {str(path): digest(path) for path in required}
    grant_path = evidence / 'gpu8-routing-source-precision-grant-v1.json'
    if options.prepare:
        payload = {'authorization': 'explicit user conditional independent SOURCE-only grant, W4 terminal/drain/lock free and Routing original source/native3/provider/post/drain all0 verified',
            'module': 'routing-quant', 'GPU': 8, 'CPU_affinity': '64-79', 'scope': 'B8/ISL4096/OSL16/TP1/EP1',
            'grant_epoch': time.time(), 'start_epoch': time.time() + 5, 'prior_owner_drain_sha256': digest(drain_path),
            'source_revision': '4cb5e929f84ebcebf98ec6702a26f1ee93190828', 'dependency_pin': 'e87870bb7aacf83bfb3be9da685f148316b8df23',
            'bindings_sha256': hashes, 'release': True, 'performance': False, 'PORT_AFTER': False, 'official_v4_wave': False}
        grant_path.write_text(json.dumps(payload, indent=2) + '\n')
    else:
        payload = json.loads(grant_path.read_text())
        assert payload['bindings_sha256'] == hashes
        packages = {}
        for kind in ['historical', 'fresh']:
            assert int((evidence / f'package-{kind}.exit').read_text()) == 0
            path = evidence / f'package-{kind}.json'
            metadata = json.loads(path.read_text())
            assert not metadata['GPU_initialized']
            packages[kind] = {'check_sha256': digest(path), 'site': metadata['site'], 'mapped_native_sha256': metadata['mapped_native_sha256'], 'package_files_sha256': metadata['package_files_sha256']}
        payload['packages'] = packages
        grant_path.write_text(json.dumps(payload, indent=2) + '\n')
        time.sleep(max(0, payload['start_epoch'] - time.time()))
    print(json.dumps({'grant': str(grant_path), 'grant_epoch': payload['grant_epoch'], 'start_epoch': payload['start_epoch']}))


if __name__ == '__main__':
    main()

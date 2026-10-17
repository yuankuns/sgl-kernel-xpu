import ast
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
PR = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    assert len(candidate['files']) == 660
    for name, record in candidate['files'].items():
        assert not name.startswith('build-xe35/')
        assert digest(PR / name) == record['sha256'], name
    assert candidate['source_digest'] == 'bce898956487b009425edb479cf26ba7e926a67572fc57943d836a8b0f0931be'
    raw = ART / 'v5-model51-gpu10-preparation/normal-v5-gpu10-accuracy.json'
    assert digest(raw) == '90fbd175fbb525799e77df8e943714c7d5f3c4b20ad9c3d8b4c4c54cd19d1643'
    for name in ('finalseal-baseline-CPU-v1.exit', 'performance-plan-CPU-v2.exit'):
        assert (ROOT / name).read_text().strip() == '0'
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    old_pids = (3265503, 3265506, 3268145, 3268072)
    assert all(not Path(f'/proc/{pid}').exists() for pid in old_pids)
    locks = Path('/proc/locks').read_text().splitlines()
    holders = {}
    for gpu in (5, 10):
        inode = os.stat(f'/tmp/intel-gpu-{gpu}.lock').st_ino
        holders[str(gpu)] = [line for line in locks if len(line.split()) > 5 and line.split()[5].endswith(f':{inode}')]
    assert not holders['10']
    report = {'epoch': time.time(), 'actual_exit': 0, 'source660_current_SHA_matches_freeze': True,
        'source_digest': candidate['source_digest'], 'GPU10_precision_raw_unchanged_SHA': digest(raw),
        'CPU_only': True, 'GPU_compile_started': False, 'old_runtime_pids_absent': list(old_pids),
        'host_gpu_lock_holders_snapshot': holders, 'FM_GPU5_logical_lease_retained': True,
        'new_performance_grant_issued': False, 'NV_handoff_authorized': False,
        'actual_current_preparation_file_SHA': {path.name: digest(path) for path in sorted(ROOT.iterdir())
            if path.is_file() and path.suffix in ('.py', '.md', '.json', '.log', '.exit')
            and path.name != 'CPU_READY_SOURCE_POSTVERIFY.json'
            and not path.name.startswith('ready-source-postverify-CPU-')}}
    (ROOT / 'CPU_READY_SOURCE_POSTVERIFY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'source660': 'unchanged', 'old_GPU_runtime': 'drained',
        'host_GPU5_lock_holders': holders['5'], 'GPU_compile_started': False,
        'performance_driver_SHA': digest(ROOT / 'performance_driver.py')}))


if __name__ == '__main__':
    main()

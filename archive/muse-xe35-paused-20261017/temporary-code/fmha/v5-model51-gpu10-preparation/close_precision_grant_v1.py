import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary_path = ROOT / 'normal-v5-gpu10-summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['GPU10_runtime_drained_lock_released']
    assert not summary['live_GPU10_runtime_processes'] and not summary['GPU10_lock_holders']
    stage = ART / 'canonical-v5-icpx'
    grant_path = stage / 'gpu10-precision-only-barrier-v1.json'
    issued_path = stage / 'gpu10-precision-only-barrier-v1-issued.json'
    before = digest(issued_path)
    issued = json.loads(issued_path.read_text())
    current = json.loads(grant_path.read_text())
    assert current == issued and current['coordinator_release'] and current['gpu'] == 10
    assert not current['closed'] and current['canonical_manifest_sha256'] == summary['current_manifest_sha256']
    current.update({'closed': True, 'closed_epoch': time.time(), 'coordinator_release': False,
        'runtime_reuse_permitted': False, 'actual_accuracy_exit': summary['actual_exit'],
        'GPU10_runtime_drained_lock_released': True, 'summary_sha256': digest(summary_path),
        'FM_GPU5_performance_lease_retained': True, 'NV_handoff_authorized': False})
    grant_path.write_text(json.dumps(current, indent=2) + '\n')
    assert digest(issued_path) == before
    print(json.dumps({'closed': True, 'issued_grant_bytes_unchanged': True,
        'issued_sha256': before, 'actual_accuracy_exit': summary['actual_exit'],
        'GPU10_runtime_drained': True, 'GPU5_performance_not_authorized': True}), flush=True)


if __name__ == '__main__':
    main()

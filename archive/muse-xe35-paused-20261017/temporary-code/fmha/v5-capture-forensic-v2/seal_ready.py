import json
from pathlib import Path

import forensic_runner as forensic


ROOT = Path(__file__).resolve().parent


def main():
    previous = ROOT / 'CPU_READY_IMMUTABLE_FREEZE_v2.json'
    report = json.loads(previous.read_text())
    assert report['actual_exit'] == 0 and report['CPU_READY'] is True
    assert (ROOT / 'CPU_READY_IMMUTABLE_FREEZE_v2.exit').read_text().strip() == '0'
    for name, checksum in report['file_SHA'].items():
        if name != 'CPU_READY_IMMUTABLE_FREEZE_v2.log':
            assert forensic.digest(ROOT / name) == checksum
    report['previous_CPU_proof_SHA'] = forensic.digest(previous)
    report['final_settled_log_seal'] = True
    report['supersedes_inflight_own_stdout_hash_in_previous_manifest'] = True
    excluded = {'CPU_READY_FINAL_SEAL_v2.json', 'CPU_READY_FINAL_SEAL_v2.log', 'CPU_READY_FINAL_SEAL_v2.exit'}
    report['file_SHA'] = {path.name: forensic.digest(path) for path in sorted(ROOT.iterdir())
        if path.is_file() and path.name not in excluded}
    with (ROOT / 'CPU_READY_FINAL_SEAL_v2.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'CPU_READY': True, 'final_settled_log_seal': True,
        'GPU_started': False, 'file_count': len(report['file_SHA'])}))


if __name__ == '__main__':
    main()

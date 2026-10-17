import fcntl
import hashlib
import json
from pathlib import Path
import re
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    evidence = directory.parent / 'historical-fresh-source-precision-v1'
    for name in ['host-launch.exit', 'gpu-runtime.exit', 'source-paired-precision.exit',
                 'original-tests-historical.exit', 'original-tests-fresh.exit']:
        assert int((evidence / name).read_text()) == 0, name
    pair = json.loads((evidence / 'source-paired-precision.json').read_text())
    assert pair['original_precision_pass'] and len(pair['shapes']) == 6
    for round_id, kind in [(0, 'historical'), (1, 'fresh'), (2, 'fresh'), (3, 'historical')]:
        assert int((evidence / f'round-{round_id}-{kind}.exit').read_text()) == 0
        assert len(json.loads((evidence / f'round-{round_id}-{kind}.json').read_text())) == 6
    tests = {}
    for kind in ['historical', 'fresh']:
        assert int((directory / f'package-post-{kind}.exit').read_text()) == 0
        before = json.loads((evidence / f'package-{kind}.json').read_text())
        after = json.loads((directory / f'package-post-{kind}.json').read_text())
        for field in ['site', 'package_files_sha256', 'mapped_native_sha256', 'provenance_sha256']:
            assert before[field] == after[field], field
        assert not after['GPU_initialized']
        log = (evidence / f'original-tests-{kind}.log').read_text()
        assert 'collected 1300 items' in log
        summary = next(line for line in reversed(log.splitlines()) if ' passed' in line)
        assert not re.search(r'\b\d+ (?:failed|error|errors|deselected)\b', summary)
        passed = re.search(r'(\d+) passed', summary)
        skipped = re.search(r'(\d+) skipped', summary)
        assert passed and int(passed[1]) + (int(skipped[1]) if skipped else 0) == 1300
        tests[kind] = {'summary': summary, 'collected': 1300, 'passed': int(passed[1]),
                       'original_skipped': int(skipped[1]) if skipped else 0, 'actual_exit': 0,
                       'post_package_sha256': digest(directory / f'package-post-{kind}.json')}
    known = [3044581, 3044585, 3046276]
    assert all(not Path(f'/proc/{pid}').exists() for pid in known)
    with Path('/tmp/intel-gpu-8.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    report = {'verified_epoch': time.time(), 'previous_owner': 'Routing SOURCE handle24419', 'GPU': 8,
              'actual_runtime_and_host_exit': 0, 'six_production_shapes_four_rounds_original_precision_pass': True,
              'pair_summary_sha256': digest(evidence / 'source-paired-precision.json'), 'original_tests': tests,
              'normal_SOURCE_sites_and_mapped_native_post_unchanged': True, 'known_owner_PIDs_absent': known,
              'gpu8_lock_nonblocking_probe_free': True, 'GPU8_released': True,
              'performance_run': False, 'PORT_AFTER_compared': False, 'new_GPU_successor_authorized': False}
    (directory / 'GPU8-actual-drain.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

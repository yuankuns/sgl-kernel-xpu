import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    grant_path = directory / 'gpu11-official-after-accuracy-grant-v2.json'
    grant = json.loads(grant_path.read_text())
    for name in ['gpu-runtime.exit', 'host-launch.exit', 'source-after-paired-precision.exit', 'original-40-after.exit',
                 'normal-package-postcheck.exit', 'live-runtime-provenance.exit']:
        assert int((directory / name).read_text()) == 0, name
    subprocess.run([sys.executable, str(directory / 'prepare_grant.py'), '--verify'], check=True)
    pair = json.loads((directory / 'source-after-paired-precision.json').read_text())
    assert pair['strict005_and_full_bitwise_pass'] and len(pair['shapes']) == 5
    references = []
    for shape, record in pair['shapes'].items():
        assert len(record['full_element_comparisons']) == 4
        for round_id, comparison in record['full_element_comparisons'].items():
            assert comparison['original_strict005_pass'] and comparison['independent_reference_pass']
            assert comparison['violating_elements'] == 0 and comparison['bitwise_different_elements'] == 0
            assert comparison['full_hash_equal'] and comparison['checked_elements'] > 0
            actual = record['AFTER_rounds'][int(round_id)]
            oracle = actual['independent_FP64']
            assert oracle['atol'] == 0.005 and oracle['rtol'] == 0.005 and oracle['max_rows_per_expert'] == 4
            assert actual['inputs'] == record['fresh_SOURCE']['inputs']
            assert actual['full_valid_output'] == record['fresh_SOURCE']['full_valid_output']
            assert actual['actual_registered_API'] == 'moe_grouped_mm_nt_mxfp4_w4a4_xe35'
            assert actual['normal_manifest_sha256'] == grant['official_manifest_sha256']
            assert actual['performance'] == 'not run'
            references.append({'shape': shape, 'round': int(round_id), 'actual_FP64_checked_elements': oracle['checked']})
    tests = (directory / 'original-40-after.log').read_text()
    assert re.search(r'\b40 passed\b', tests) and not re.search(r'\b\d+ (?:failed|skipped|xfailed|deselected)\b', tests)
    before = json.loads((directory / 'normal-package-check.json').read_text())
    after = json.loads((directory / 'normal-package-postcheck.json').read_text())
    for name in ['manifest_sha256', 'current_native_count', 'mapped_native_hashes', 'source_digest', 'new_ops', 'compiler_policy']:
        assert before[name] == after[name], name
    assert not after['xpu_initialized']
    known = [3080482, 3080592, 3080884, 3081500]
    assert all(not Path(f'/proc/{pid}').exists() for pid in known)
    live = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if directory.name in command and any(name in command for name in ['run_accuracy_v2.sh', 'launch_accuracy_v2.sh', 'after_accuracy_only.py', 'pytest']):
            live.append({'pid': int(entry.name), 'command': command})
    assert not live, live
    with Path('/tmp/intel-gpu-11.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    report = {'verified_epoch': time.time(), 'true_gpu_runtime_exit': 0, 'true_host_launch_exit': 0,
              'source_after_strict005_and_full_bitwise_pass': True, 'production_shapes': 5, 'AFTER_rounds': 4,
              'original_tests_passed': 40, 'FP64_scope': 'max_rows4/expert/allcolumns; not every production row oracle',
              'actual_FP64_reference_counts': references, 'official_manifest_sha256': grant['official_manifest_sha256'],
              'pair_summary_sha256': digest(directory / 'source-after-paired-precision.json'), 'grant_sha256': digest(grant_path),
              'protected_source_provider_and_native_postcheck': True, 'known_host_PIDs_absent': known,
              'matching_runtime_processes': live, 'gpu11_lock_nonblocking_probe_free': True, 'GPU11_released': True,
              'performance_run': False, 'performance_authorized': False, 'final_two_port_acceptance': False}
    (directory / 'GPU11-actual-drain.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

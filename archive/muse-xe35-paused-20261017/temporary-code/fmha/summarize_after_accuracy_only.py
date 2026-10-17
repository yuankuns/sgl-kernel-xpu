import hashlib
import json
import os
from pathlib import Path
import re
import time


ART = Path(__file__).resolve().parent
LABEL = 'normal-v4-gpu5-accuracy-only-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = json.loads((ART / f'{LABEL}.json').read_text())
    actual_exit = int((ART / f'{LABEL}.exit').read_text().strip())
    handle = (ART / f'{LABEL}.handle').read_text()
    supervisor = int(re.search(r'supervisor_pid=(\d+)', handle).group(1))
    live = []
    for pid in (supervisor, 3120853):
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and (b'after_accuracy_only' in command.read_bytes() or LABEL.encode() in command.read_bytes()):
            live.append(pid)
    for command in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = command.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(os.fsdecode(argument)).name == 'after_accuracy_only.py' for argument in arguments if argument):
            live.append(int(command.parent.name))
    lock = Path('/tmp/intel-gpu-5.lock').stat()
    lock_key = f'{os.major(lock.st_dev):02x}:{os.minor(lock.st_dev):02x}:{lock.st_ino}'
    active_locks = [line for line in Path('/proc/locks').read_text().splitlines() if lock_key in line]
    assert live == [] and active_locks == [], (live, active_locks)
    records = raw['cases']
    source = json.loads((ART / 'source-only-v1-fresh-optimized.json').read_text())
    source_records = {record['name']: record for record in source['cases']}
    assert len(records) == 38
    assert sum(record['name'].startswith('prefill') for record in records) == 6
    assert all(record['input_sha256'] == source_records[record['name']]['input_sha256'] for record in records)
    assert all(record['coverage'] == 'full' and record['atol'] == record['rtol'] == 5e-3 for record in records)
    checked = sum(record['checked_elements'] for record in records)
    assert checked == 1075838976
    reference_count = sum(record['reference_PASS'] for record in records)
    bitwise_count = sum(record['full_output_bitwise'] for record in records)
    grant_path = ART / 'canonical-v4-icpx/gpu5-accuracy-only-barrier-v1.json'
    issued = grant_path.with_name(grant_path.stem + '-issued.json')
    assert digest(issued) == raw['grant_sha256']
    report = {'epoch': time.time(), 'canonical_version': raw['canonical_version'], 'exit': actual_exit,
              'scope': raw['scope'], 'GPU': 5, 'CPU': '16-31', 'cases': len(records),
              'runtime_host_pid': 3120853, 'runtime_container_pid': raw['runtime_pid'], 'supervisor_host_pid': supervisor,
              'full_prefill_cases': 6, 'full_decode_cases': 32, 'checked_elements': checked,
              'reference_pass_cases': reference_count, 'full_output_bitwise_cases': bitwise_count,
              'all_input_SHA_equal_to_fresh_SOURCE': True, 'atol': 5e-3, 'rtol': 5e-3,
              'max_abs_to_independent_CPU_reference': max(record['max_abs'] for record in records),
              'max_abs_to_fresh_SOURCE': max(record['source_port_max_abs'] for record in records),
              'mapped_native_count': raw['mapped_native_count'], 'canonical_manifest_sha256': raw['canonical_manifest_sha256'],
              'source_digest': raw['source_digest'], 'compiler_policy_sha256': raw['compiler_policy_sha256'],
              'grant_sha256': raw['grant_sha256'], 'raw_JSON_sha256': digest(ART / f'{LABEL}.json'),
              'raw_log_sha256': digest(ART / f'{LABEL}.log'), 'handle_sha256': digest(ART / f'{LABEL}.handle'),
              'required_native_sha256': raw['required_native_sha256'], 'GPU_lock_released': True,
              'live_runtime_processes': live, 'no_timing_warmup_events_graphs': True,
              'official_performance_wave': False, 'logical_GPU5_lease': 'FMHA retained; NV5 requires main explicit grant',
              'production_model_wave_coverage': 'not claimed; actual3/5 mapped CPU-only and frozen optimized guard falls back',
              'source_port_and_true_preopt_port_BAAB': 'pending new main formal performance grant'}
    (ART / 'normal-v4-gpu5-accuracy-only-v1-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    grant = json.loads(grant_path.read_text())
    grant.update({'coordinator_release': False, 'completion_epoch': time.time(), 'runtime_exit': actual_exit,
                  'GPU_lock_released': True, 'reuse_permitted': False})
    grant_path.write_text(json.dumps(grant, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if actual_exit != 0 or reference_count != 38 or bitwise_count != 38:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

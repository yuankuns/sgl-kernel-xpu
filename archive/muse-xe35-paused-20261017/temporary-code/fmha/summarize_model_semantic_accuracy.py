import hashlib
import json
import os
from pathlib import Path
import time


ART = Path(__file__).resolve().parent
ROOT = ART / 'model-config51-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    suite_exit = int((ART / 'model51-accuracy-suite-v1.exit').read_text())
    live = []
    for command in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = command.read_bytes().split(b'\0')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(os.fsdecode(argument)).name == 'model_semantic_accuracy.py' for argument in arguments if argument):
            live.append(int(command.parent.name))
    lock = Path('/tmp/intel-gpu-5.lock').stat()
    lock_key = f'{os.major(lock.st_dev):02x}:{os.minor(lock.st_dev):02x}:{lock.st_ino}'
    locks = [line for line in Path('/proc/locks').read_text().splitlines() if lock_key in line]
    assert live == [] and locks == []
    plan = json.loads((ROOT / 'plan.json').read_text())
    providers = {}
    raws = {}
    for provider in ('source', 'port'):
        exit_file = ART / f'model51-accuracy-{provider}-v1.exit'
        raw_file = ROOT / f'{provider}-accuracy.json'
        if not exit_file.exists():
            providers[provider] = {'started': False, 'reason': 'previous provider failure; no substitute API/kernel'}
            continue
        actual_exit = int(exit_file.read_text())
        if not raw_file.exists():
            providers[provider] = {'started': True, 'exit': actual_exit, 'cases': 0, 'failure': 'import or preflight before first native call; raw log retained'}
            continue
        raw = json.loads(raw_file.read_text())
        raws[provider] = raw
        records = raw['cases']
        for record in records:
            if 'saved_full_output_file_sha256' in record:
                path = ROOT / f'{provider}-full-outputs' / f'{record["name"]}.pt'
                assert digest(path) == record['saved_full_output_file_sha256']
            if 'atol' in record:
                assert record['atol'] == record['rtol'] == 5e-3 and record['coverage'] == 'full'
        providers[provider] = {'started': True, 'exit': actual_exit, 'cases': len(records),
            'full_reference_pass_cases': sum(record.get('reference_PASS', False) for record in records),
            'checked_elements': sum(record.get('checked_elements', 0) for record in records),
            'full_output_saved_cases': sum('saved_full_output_file_sha256' in record for record in records),
            'mapped_native_count': raw['mapped_native_count'], 'raw_JSON_sha256': digest(raw_file),
            'raw_log_sha256': digest(ART / f'model51-accuracy-{provider}-v1.log'),
            'failures': [record for record in records if record.get('accuracy') != 'PASS']}
        if actual_exit == 0:
            assert len(records) == 51 and providers[provider]['checked_elements'] == 808255488
    paired = None
    if 'source' in raws and 'port' in raws:
        source = {case['name']: case for case in raws['source']['cases']}
        port = raws['port']['cases']
        paired = {'shared_input_SHA_equal_cases': sum(record['input_sha256'] == source[record['name']]['input_sha256'] for record in port),
                  'full_output_bitwise_cases': sum(record.get('full_output_bitwise', False) for record in port),
                  'full_output_nonbitwise_cases': [record['name'] for record in port if record.get('full_output_bitwise') is False],
                  'max_abs': max((record.get('source_port_max_abs', 0.0) for record in port), default=None)}
    issued = ROOT / 'gpu5-accuracy-only-barrier-v1-issued.json'
    report = {'epoch': time.time(), 'suite_exit': suite_exit, 'workload': plan['workload'],
              'providers': providers, 'paired': paired, 'GPU': 5, 'CPU': '16-31',
              'plan_sha256': digest(ROOT / 'plan.json'), 'grant_sha256': digest(issued),
              'port_route': 'all51 genuine generic fallback: actual backend omits max_seqlen_k, normal wrapper keeps0',
              'no_timing_warmup_events_graphs': True, 'GPU_lock_released': True, 'live_runtime_processes': [],
              'logical_GPU5_lease': 'FMHA retained; no NV handoff or performance grant',
              'source_kernel_guard_math_changed': False, 'production_claim': 'config/call semantic precision; not raw activation replay or optimized real-model/performance acceptance'}
    report['root_owned_grant_revocation'] = 'separate wrapper CPU metadata-only step after host drain audit'
    (ART / 'model51-accuracy-suite-v1-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if suite_exit != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

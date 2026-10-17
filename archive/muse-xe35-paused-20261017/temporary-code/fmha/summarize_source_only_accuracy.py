import hashlib
import json
import os
from pathlib import Path
import time


ART = Path(__file__).resolve().parent
ROOT = ART / 'matched-native-icpx-v1'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    expected_cases = [f'prefill-wave{requests}-w{window}' for requests in (1, 7, 8) for window in (-1, 2047)]
    expected_cases += [f'decode-osl{step}-w{window}' for step in range(1, 17) for window in (-1, 2047)]
    results = {}
    for provider in ('historical-optimized', 'fresh-optimized', 'fresh-preopt'):
        prefix = ART / f'source-only-v1-{provider}'
        data = json.loads(prefix.with_suffix('.json').read_text())
        assert prefix.with_suffix('.exit').read_text().strip() == '0'
        assert [record['name'] for record in data['cases']] == expected_cases
        assert data['all_38_reference_PASS'] is True
        for record in data['cases']:
            assert record['accuracy'] == 'PASS' and record['reference_PASS'] is True
            assert record['failed_chunks'] == 0 and record['first_failure'] is None
            assert record['coverage'] == 'full' and record['query_positions'] == 'all'
            assert record['heads_channels_requests'] == 'all'
            assert record['atol'] == record['rtol'] == 5e-3
            assert record['checked_elements'] == record['requests'] * record['query_length'] * 128 * 64
            assert record['native_launches'] == 1
            assert all(name not in record for name in ('samples_ms', 'median_ms', 'warmup', 'graph_replays'))
        results[provider] = {'exit': 0, 'reference_cases': 38, 'prefill_full_all_query_cases': 6,
            'decode_full_reference_cases': 32, 'checked_elements': sum(record['checked_elements'] for record in data['cases']),
            'max_abs': max(record['max_abs'] for record in data['cases']),
            'runtime_JSON_sha256': digest(prefix.with_suffix('.json')), 'raw_log_sha256': digest(prefix.with_suffix('.log')),
            'runtime_provenance': data['actual_runtime_provenance'], 'native_compiler_provenance': data['compiler_provenance']}
    historical = json.loads((ART / 'source-only-v1-historical-optimized.json').read_text())
    fresh = json.loads((ART / 'source-only-v1-fresh-optimized.json').read_text())
    for original, current in zip(historical['cases'], fresh['cases'], strict=True):
        assert current['full_output_bitwise'] is True
        assert original['input_sha256'] == current['input_sha256']
        assert original['output_sha256'] == current['output_sha256']
    assert (ART / 'source-only-gpu5-suite-v1.exit').read_text().strip() == '0'
    live = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            name = (directory / 'comm').read_text().strip()
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if name.startswith('python') and 'source_only_accuracy.py --provider' in command:
                live.append({'pid': int(directory.name), 'command': command})
        except OSError:
            pass
    assert not live, live
    stat = Path('/tmp/intel-gpu-5.lock').stat()
    marker = f'{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
    lock_records = [line for line in Path('/proc/locks').read_text().splitlines() if marker in line]
    assert not lock_records, lock_records
    report = {'epoch': time.time(), 'GPU': 5, 'CPU': '16-31', 'scope': 'SOURCE-only accuracy, not PORT or official wave',
        'workload': 'B8 ISL4096 OSL16 TP1 EP1 BF16 Hq128 Hkv16 D64 page64', 'atol': 5e-3, 'rtol': 5e-3,
        'providers': results, 'all_provider_exits': 0, 'supervisor_exit': 0,
        'historical_to_fresh_optimized_all_inputs_byte_SHA_equal': True,
        'historical_to_fresh_optimized_full_output_bitwise_cases': 38,
        'no_perf_warmup_events_graphs_PORT': True, 'GPU_lock_released': True, 'live_runtime_processes': live,
        'logical_GPU5_lease': 'FMHA retained; NV5 requires separate main grant',
        'official_v4_source_port_and_preopt_port_BAAB': 'pending fresh main official wave grant',
        'precision_algorithm_or_dispatch_changed': False,
        'confound_boundary': 'old optimized native icx/GNU to fresh icpx bitwise equal for all38; fresh preopt/source native icpx but historical non-target package/e878/profiling ON differ from PORT public7b56/profiling OFF'}
    (ROOT / 'source-only-gpu5-accuracy-v1-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_providers_full38_PASS': True, 'optimized_bitwise38': True, 'GPU5_lock_released': True,
                      'official_wave': 'pending', 'summary': str(ROOT / 'source-only-gpu5-accuracy-v1-summary.json')}))


if __name__ == '__main__':
    main()

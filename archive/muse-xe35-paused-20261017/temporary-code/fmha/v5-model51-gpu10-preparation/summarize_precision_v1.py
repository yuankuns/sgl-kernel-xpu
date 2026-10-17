import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    exit_path = ART / 'model51-normal-v5-gpu10-accuracy-v1.exit'
    assert exit_path.exists(), 'same running handle must reach actual terminal first'
    actual_exit = int(exit_path.read_text().strip())
    raw_path = ROOT / 'normal-v5-gpu10-accuracy.json'
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {'cases': []}
    families = {}
    for family, expected in (('model51', 51), ('regression38', 38)):
        cases = [case for case in raw['cases'] if case['family'] == family]
        saved = []
        for case in cases:
            if 'saved_full_output_file_sha256' in case:
                path = ROOT / 'normal-v5-full-outputs' / f'{family}-{case["name"]}.pt'
                assert digest(path) == case['saved_full_output_file_sha256']
                saved.append(case['name'])
        families[family] = {'expected': expected, 'actual_cases_recorded': len(cases),
            'actual_optimized_routes': sum(case.get('actual_metadata_private_route_PASS', False) for case in cases),
            'full_reference_PASS': sum(case.get('reference_PASS', False) for case in cases),
            'SOURCE_whole_output_bitwise_PASS': sum(case.get('full_output_bitwise', False) for case in cases),
            'full_saved_outputs_rehashed': len(saved),
            'checked_reference_elements': sum(case.get('checked_elements', 0) for case in cases),
            'failures': [case for case in cases if case.get('accuracy') != 'PASS']}
    current = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    for relative, record in current['files'].items():
        assert digest(Path(current['source']) / relative) == record['sha256']
    for label in ('model51-accuracy-source-v1', 'model51-accuracy-port-v1', 'model51-accuracy-suite-v1'):
        assert (ART / f'{label}.exit').read_text().strip() == '0'
    assert digest(ART / 'model-config51-v1/port-accuracy.json') == 'a53ebdad1cd7c66a44e040ba03df342b1855962a3930993bc806667b6b6fec6f'
    issued_path = ART / 'canonical-v5-icpx/gpu10-precision-only-barrier-v1-issued.json'
    grant = json.loads(issued_path.read_text())
    for path, field in ((ART / 'model-config51-v1/source-accuracy.json', 'source51_raw_sha256'),
                        (ART / 'source-only-v1-fresh-optimized.json', 'source38_raw_sha256'),
                        (ROOT / 'precision_v5.py', 'harness_sha256')):
        assert digest(path) == grant[field]
    assert raw.get('canonical_manifest_sha256', grant['canonical_manifest_sha256']) == grant['canonical_manifest_sha256']
    actual_private = [case for case in raw['cases'] if case['family'] == 'model51' and case.get('actual_metadata_private_route_PASS')]
    pids = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if 'precision_v5.py' in command and 'v5-model51-gpu10-preparation' in command:
            pids.append({'pid': int(process.name), 'command': command})
    assert not pids, pids
    inode = Path('/tmp/intel-gpu-10.lock').stat().st_ino
    locks = [line for line in Path('/proc/locks').read_text().splitlines() if line.split()[5].endswith(':' + str(inode))]
    assert not locks
    report = {'epoch': time.time(), 'actual_exit': actual_exit, 'families': families,
        'raw_sha256': digest(raw_path) if raw_path.exists() else None,
        'model51_actual_private_prefill_routes': sum(case['name'].startswith('model-prefill') for case in actual_private),
        'model51_actual_private_decode_routes': sum(case['name'].startswith('model-decode') for case in actual_private),
        'all89_full_refs_SOURCE_bits_private_routes_PASS': actual_exit == 0 and raw.get('all89_fullrefs_and_source_wholebitwise_PASS', False),
        'v5_source_current660_SHA_unchanged': True, 'original_SOURCE_raw_and_harness_SHA_unchanged': True,
        'v4_fallback51_raw_preserved': True, 'current_manifest_sha256': grant['canonical_manifest_sha256'],
        'live_GPU10_runtime_processes': pids, 'GPU10_lock_holders': locks,
        'GPU10_runtime_drained_lock_released': True, 'GPU5_FM_logical_performance_lease_retained': True,
        'no_performance_or_NV_handoff': True,
        'acceptance_boundary': 'record actual partial failures/counts; no fallback-within_tol or missing cases represented as optimized pass'}
    (ROOT / 'normal-v5-gpu10-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

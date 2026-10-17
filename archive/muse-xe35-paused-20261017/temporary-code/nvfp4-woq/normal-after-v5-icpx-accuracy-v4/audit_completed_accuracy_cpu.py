import fcntl
import json
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from process_tree import ProcessTree, snapshot
from providers import DIRECTORY, FRESH, SITES, SOURCE_ACCURACY, digest
from supervisor import provenance


def host_path(path):
    return Path(str(path).replace('/root/', '/home/syk/', 1))


def main():
    ready_path = DIRECTORY / 'AFTER_ACCURACY_READY.json'
    ready = json.loads(ready_path.read_text())
    post = json.loads((DIRECTORY / 'postcheck-CPU.json').read_text())
    grant = json.loads((DIRECTORY / 'main-grant.json').read_text())
    current = provenance()
    assert post['provenance'] == current
    assert ready['grant_sha256'] == digest(DIRECTORY / 'main-grant.json')
    assert current['scripts_sha256'] == grant['scripts_sha256']
    assert current['canonical_manifest_sha256'] == ready['current_after_manifest_sha256']
    for name, checksum in ready['raw_sha256'].items():
        assert digest(DIRECTORY / name) == checksum, 'Sealed evidence changed: ' + name
    phases = ('source', 'after', 'after', 'source')
    verified = {}
    shapes = {'tails': {(4, 11, 192, 64), (3, 65, 256, 128)}, 'production': {(256, rows, 8192, inner) for rows in (64, 262144) for inner in (8192, 4096)}}
    for family in ('tails', 'production'):
        baseline = {}
        cases = []
        for phase, label in enumerate(phases):
            report = json.loads((DIRECTORY / 'results' / family / f'round0-phase{phase}-{label}.json').read_text())
            assert (DIRECTORY / f'{family}-p{phase}-{label}.exit').read_text().strip() == '0'
            records = report['results']
            assert len(records) == 4
            assert {tuple(record['shape']) for record in records} == shapes[family]
            for record in records:
                key = (tuple(record['shape']), record['mode'])
                assert record['label'] == label and record['phase'] == phase and record['round'] == 0
                assert record['samples_us'] == []
                assert record['mode'] in (('bf16', 'mxfp8') if family == 'tails' else ('bf16',))
                counts = record['counts']
                assert sum(counts) == record['shape'][1] and len(counts) == record['shape'][0]
                maximum = max(counts) if family == 'tails' else 3
                coverage = record['independent_reference_coverage']
                covered = [min(count, maximum) for count in counts]
                assert coverage['compute_dtype'] == 'CPU FP32'
                assert coverage['max_rows_per_expert'] == maximum
                assert coverage['direct_in_this_phase'] == (phase in (0, 1))
                assert coverage['oracle_rows_per_expert'] == covered
                assert coverage['uncovered_rows_per_expert'] == [count - rows for count, rows in zip(counts, covered)]
                if phase in (0, 1):
                    reference = record['reference']
                    assert reference['reference'] == 'CPU FP32 matmul of independently decoded production operands'
                    assert reference['checked'] == sum(covered) * record['shape'][2]
                    assert reference['atol'] == 0.005 and reference['rtol'] == 0.005
                else:
                    assert record['reference']['direct_reference'] is False
                components = {'output': record['output'], **record['scales']}
                assert set(components) == {'output', 'c', 'sf'}
                for evidence in components.values():
                    assert evidence['exact'] and evidence['all_elements_checked'] and evidence['max_abs'] == 0
                    assert evidence['first_phase_cache_created'] == (phase == 0)
                signature = {'inputs': record['inputs'], 'counts': counts, 'bit_hashes': {name: evidence['sha256'] for name, evidence in components.items()}}
                if phase == 0:
                    assert key not in baseline
                    baseline[key] = signature
                else:
                    assert baseline[key] == signature, 'Whole valid bits/input hashes differ across phases'
                cases.append({'phase': phase, 'label': label, 'shape': record['shape'], 'mode': record['mode'], 'bit_hashes': signature['bit_hashes'], 'direct_reference': phase in (0, 1), 'independent_FP32_elements': sum(covered) * record['shape'][2] if phase in (0, 1) else 0, 'uncovered_rows': sum(counts) - sum(covered), 'all_valid_bits_equal': True})
        assert len(baseline) == 4 and len(cases) == 16
        verified[family] = cases
    cases = ET.parse(DIRECTORY / 'original-tests-after.xml')
    assert len(cases.findall('.//testcase')) == 50
    assert all(not cases.findall('.//' + kind) for kind in ('skipped', 'failure', 'error'))
    assert (DIRECTORY / 'original-tests-after.exit').read_text().strip() == '0'
    maps = {}
    for label in ('source', 'after'):
        provider = post['providers'][label]
        assert not provider['GPU_initialized']
        site = SITES[label].resolve()
        entries = {}
        for name, evidence in provider['native_mappings'].items():
            path = host_path(evidence['path']).resolve()
            assert path.is_relative_to(site) and path.name == name
            assert digest(path) == evidence['sha256']
            entries[name] = evidence['sha256']
        maps[label] = entries
    assert len(maps['after']) == 714
    source_manifest = json.loads((FRESH / 'source-binary-manifest.json').read_text())
    assert {str(path.relative_to(SITES['source'])): digest(path) for path in SITES['source'].rglob('*') if path.is_file()} == source_manifest['package_files_sha256']
    drains = {}
    for directory in (SOURCE_ACCURACY, DIRECTORY):
        for path in directory.glob('*.owned.json'):
            record = json.loads(path.read_text())
            metrics = ProcessTree(0, record['token'], known=record['identities']).sample()
            assert not metrics['live']
            drains[str(path)] = {'PID_startticks_drained': True, 'identities_checked': len(record['identities'])}
    waiter = json.loads((DIRECTORY / 'waiter-ready.json').read_text())
    process = snapshot().get(waiter['PID'])
    assert process is None or process['start_ticks'] != int(waiter['startticks']) or process['state'] in ('Z', 'X')
    for name in ('host-supervisor.exit', 'GPU-wrapper.exit', 'postcheck-CPU.exit', 'CPU-proof-wrapper.exit'):
        assert (DIRECTORY / name).read_text().strip() == '0'
    with Path('/tmp/intel-gpu-9.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stack = json.loads((DIRECTORY / 'runtime-stack-snapshot-v1.json').read_text())
    current_stack = {'kernel_release': platform.release(), 'loaded_xe_srcversion': Path('/sys/module/xe/srcversion').read_text().strip(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    assert current_stack['kernel_release'] == stack['kernel_running']['release']
    assert current_stack['loaded_xe_srcversion'] == stack['loaded_xe_srcversion']
    assert current_stack['boot_id'] == stack['boot_id']
    assert 'torch' not in sys.modules
    result = {'actual_independent_CPU_audit_exit': 0, 'epoch': time.time(), 'canonical_version': 'canonical-v5-icpx', 'driver_version': 'accuracy-v4', 'READY_sha256': digest(ready_path), 'provenance': current, 'existing_raw_seals_rechecked': len(ready['raw_sha256']), 'record_to_record_full_valid_output_and_c_sf_bit_hashes_equal': True, 'reference_scope': 'CPU FP32: production direct phases0/1 max3rows/expert; tails all rows; not FP64/not all production independently oracled', 'verified_records': verified, 'original_pytest_passes': 50, 'original_pytest_skips': 0, 'static_native_map_file_seals': maps, 'current_normal_package_import_evidence': 'Existing genuine CPU proof/post providers actual0; their unchanged package/Python/native files rehashed now, not a new import or GPU run.', 'drains': drains, 'supervisor_PID_startticks_terminal': True, 'physical_GPU9_nonblocking_lock_probe_actual_exit': 0, 'running_stack': current_stack, 'Torch_imported_by_audit': False, 'GPU_started': False, 'compiler_started': False, 'GPU5_started': False, 'performance_acceptance': False, 'full_task_acceptance': False}
    target = DIRECTORY / 'independent-completed-CPU-audit-v1.json'
    with target.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_independent_CPU_audit_exit': 0, 'sealed_raw_files': len(ready['raw_sha256']), 'native_maps_SOURCE': len(maps['source']), 'native_maps_AFTER': len(maps['after']), 'tail_records': 16, 'production_records': 16, 'physical9_free': True, 'Torch_imported': False, 'report_sha256': digest(target)}, indent=2))


if __name__ == '__main__':
    main()

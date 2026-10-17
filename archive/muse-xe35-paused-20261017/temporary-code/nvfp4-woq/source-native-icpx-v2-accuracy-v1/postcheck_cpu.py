import json
import subprocess
from pathlib import Path

from providers import DIRECTORY, PYTHONS, digest
from supervisor import environment, provenance


grant = json.loads((DIRECTORY / 'grant.json').read_text())
assert provenance() == grant['provenance']
summary = json.loads((DIRECTORY / 'summary.json').read_text())
assert summary['actual_suite_exit'] == 0 and summary['skips'] == 0
assert summary['original_tests'] == {'historical': 50, 'fresh': 50}
assert (DIRECTORY / 'host-supervisor.exit').read_text().strip() == '0'
assert (DIRECTORY / 'GPU-wrapper.exit').read_text().strip() == '0'
live = []
phases = []
for path in sorted(DIRECTORY.glob('*.drain.json')):
    if path.name.startswith(('tests-', 'tails-', 'production-')):
        data = json.loads(path.read_text())
        assert data['drained'] and not data['live_after']
        phases.append(path.name)
        for identity in data['identities']:
            try:
                fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().rpartition(')')[2].split()
            except FileNotFoundError:
                continue
            if int(fields[19]) == int(identity['start_ticks']) and fields[0] != 'Z':
                live.append(identity)
assert len(phases) == 10 and not live
suite = json.loads((DIRECTORY / 'GPU-suite-started.json').read_text())
try:
    fields = (Path('/proc') / str(suite['PID']) / 'stat').read_text().rpartition(')')[2].split()
    assert int(fields[19]) != int(suite['startticks']) or fields[0] == 'Z'
except FileNotFoundError:
    pass
for family in ('tails', 'production'):
    assert summary['families'][family]['records'] == 16
    assert summary['families'][family]['full_valid_outputs_and_scales_bitwise']
    assert summary['families'][family]['timing_samples'] == 0
    for phase, label in enumerate(('historical', 'fresh', 'fresh', 'historical')):
        data = json.loads((DIRECTORY / 'results' / family / ('round0-phase' + str(phase) + '-' + label + '.json')).read_text())
        assert len(data['results']) == 4
        for result in data['results']:
            assert result['samples_us'] == [] and result['output']['exact'] and result['output']['all_elements_checked']
            assert all(record['exact'] and record['all_elements_checked'] for record in result['scales'].values())
            assert result['independent_reference_coverage']['compute_dtype'] == 'CPU FP32'
            assert result['independent_reference_coverage']['max_rows_per_expert'] == (max(result['counts']) if family == 'tails' else 3)
providers = {}
for label in ('historical', 'fresh'):
    providers[label] = json.loads(subprocess.check_output([str(PYTHONS[label]), '-B', str(DIRECTORY / 'providers.py'), label], env=environment(label), text=True, timeout=300))
    assert not providers[label]['GPU_initialized']
print(json.dumps({'actual_postcheck_exit': 0, 'source_only': True, 'GPU_initialized': False, 'all_historical_and_fresh_package_inventory_native_SHA_reverified': True, 'source_body_helpers_flags_grf_e878_postverified': True, 'PID_startticks_live_after': live, 'GPU_suite_PID_startticks_terminal': True, 'completed_phases': phases, 'providers': providers, 'summary_sha256': digest(DIRECTORY / 'summary.json'), 'grant_sha256': digest(DIRECTORY / 'grant.json'), 'driver_sha256': digest(Path(__file__)), 'normal_AFTER_accepted': False, 'performance_accepted': False}, indent=2))

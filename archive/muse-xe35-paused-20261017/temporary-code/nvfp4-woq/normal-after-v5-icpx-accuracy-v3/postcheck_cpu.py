import json
import xml.etree.ElementTree as ET

from process_tree import ProcessTree, snapshot
from providers import DIRECTORY, SOURCE_ACCURACY, digest
from supervisor import PYTHONS, VERSION, environment, provenance, run, save, verify_grant


def check():
    grant_path = DIRECTORY / 'main-grant.json'
    grant = verify_grant(grant_path)
    assert grant['accuracy_only'] and not grant['performance_allowed']
    assert (DIRECTORY / 'host-supervisor.exit').read_text().strip() == '0'
    summary = json.loads((DIRECTORY / 'summary.json').read_text())
    assert summary['actual_suite_exit'] == 0 and summary['original_tests'] == 50 and summary['skips'] == 0
    assert summary['normal_AFTER_accuracy_verified'] and not summary['performance_acceptance']
    assert not summary['full_task_acceptance']
    cases = ET.parse(DIRECTORY / 'original-tests-after.xml')
    assert len(cases.findall('.//testcase')) == 50
    assert all(not cases.findall('.//' + name) for name in ('skipped', 'error', 'failure'))
    for family in ('tails', 'production'):
        assert summary['families'][family]['records'] == 16
        assert summary['families'][family]['full_valid_outputs_and_scales_bitwise']
        assert summary['families'][family]['max_abs'] == 0
        assert summary['families'][family]['timing_samples'] == 0
        for phase, label in enumerate(('source', 'after', 'after', 'source')):
            report_path = DIRECTORY / 'results' / family / f'round0-phase{phase}-{label}.json'
            report = json.loads(report_path.read_text())
            assert len(report['results']) == 4
            for result in report['results']:
                assert result['samples_us'] == []
                assert result['output']['exact'] and result['output']['all_elements_checked']
                assert all(item['exact'] and item['all_elements_checked'] for item in result['scales'].values())
                coverage = result['independent_reference_coverage']
                counts = result['counts']
                maximum = max(counts) if family == 'tails' else 3
                assert coverage['compute_dtype'] == 'CPU FP32'
                assert coverage['direct_in_this_phase'] == (phase in (0, 1))
                assert coverage['max_rows_per_expert'] == maximum
                assert coverage['oracle_rows_per_expert'] == [min(count, maximum) for count in counts]
                assert coverage['uncovered_rows_per_expert'] == [max(0, count - maximum) for count in counts]
    providers = {}
    for label in ('source', 'after'):
        name = 'post-provider-' + label
        run([str(PYTHONS[label]), '-B', str(DIRECTORY / 'providers.py'), label, '--version', VERSION], name, environment(label), timeout=900)
        providers[label] = json.loads((DIRECTORY / (name + '.log')).read_text())
        assert not providers[label]['GPU_initialized']
    drained = {}
    for directory in (SOURCE_ACCURACY, DIRECTORY):
        for path in directory.glob('*.owned.json'):
            if path.name == 'GPU-wrapper.owned.json':
                continue
            record = json.loads(path.read_text())
            metrics = ProcessTree(0, record['token'], known=record['identities']).sample()
            assert not metrics['live'], 'Owned descendant live: ' + str(path)
            drained[str(path)] = {'all_known_PID_startticks_absent_or_terminal': True, 'identities': record['identities']}
        started_path = directory / 'GPU-suite-started.json'
        started = json.loads(started_path.read_text())
        current = snapshot().get(started['PID'])
        assert current is None or current['start_ticks'] != int(started['startticks']) or current['state'] in ('Z', 'X')
        drained[str(started_path)] = {'PID': started['PID'], 'startticks': started['startticks'], 'terminal': True}
    release = json.loads((DIRECTORY / 'GPU-release.json').read_text())
    assert release['actual_wrapper_exit'] == 0 and release['all_owned_descendants_drained']
    assert release['actual_wrapper_lock_released'] and not release['GPU5_performance_granted']
    raw = {}
    for path in DIRECTORY.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log', '.exit', '.command', '.xml', '.py') and 'E' not in path.parent.name[:1] and not path.name.startswith('postcheck-CPU.'):
            raw[str(path.relative_to(DIRECTORY))] = digest(path)
    report = {'actual_postcheck_CPU_exit': 0, 'canonical_version': VERSION, 'grant_sha256': digest(grant_path), 'provenance': provenance(), 'original_pytest_passes': 50, 'pytest_skips': 0, 'tail_records': 16, 'production_records': 16, 'coverage_66_is_not_pytest_count': True, 'all_valid_output_and_scales_bitwise': True, 'oracle_scope': 'Original CPU FP32: production max3rows/active expert phase0/1; tails all rows; remaining full-output bitwise is not independent full production oracle.', 'all_known_owned_descendants_terminal': drained, 'providers': providers, 'raw_sha256': raw, 'GPU_initialized_by_postcheck': False, 'performance_acceptance': False, 'full_task_acceptance': False, 'logical_GPU5_released': False}
    save(DIRECTORY / 'postcheck-CPU.json', report)
    print(json.dumps({'actual_postcheck_CPU_exit': 0, 'GPU_initialized': False, 'original_pytest_passes': 50, 'tail_records': 16, 'production_records': 16, 'strict_bitwise': True, 'performance_acceptance': False}, indent=2))


if __name__ == '__main__':
    check()

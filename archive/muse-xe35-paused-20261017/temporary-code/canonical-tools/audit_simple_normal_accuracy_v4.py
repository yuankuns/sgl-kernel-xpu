import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    run = ROOT / 'simple-normal-accuracy-v4-run1'
    assert json.loads((run / 'complete.json').read_text())['actual_exit'] == 0
    launch = json.loads((run / 'launch-manifest.json').read_text())
    for name, expected in launch['driver_sha256'].items():
        assert digest(ROOT / name) == expected
    reports = {}
    for component, test_count, case_count in (('memory', 24, 5), ('position', 9, 4), ('logits-vocab', 36, 5)):
        evidence = ROOT / component / 'canonical-validation-v4-icpx'
        official = ROOT / component / 'canonical-v4-icpx'
        assert (run / f'{component}.exit').read_text().strip() == '0'
        grant = json.loads((run / f'{component}-grant.json').read_text())
        assert grant['gpu'] == 11 and grant['phase'] == 'accuracy'
        assert grant['serving_batch'] == 8 and grant['performance_authorized'] is False
        assert grant['package_manifest_sha256'] == digest(official / 'source-binary-manifest.json')
        assert grant['source_comparator_manifest_sha256'] == digest(evidence / 'source-comparator-manifest.json')
        provider_reports = {}
        for label in ('before', 'after'):
            for phase in ('package', 'original', 'production'):
                prefix = f'accuracy-{phase}-{label}' + ('-preflight' if phase == 'package' else '')
                assert (evidence / f'{prefix}.exit').read_text().strip() == '0'
            original_log = evidence / f'accuracy-original-{label}.log'
            text = original_log.read_text()
            assert re.search(rf'\b{test_count} passed\b', text)
            assert not re.search(r'\b\d+\s+(?:failed|skipped|xfailed|xpassed|errors?)\b', text, re.I)
            report_path = evidence / f'accuracy-production-{label}.json'
            report = json.loads(report_path.read_text())
            assert report['component'] == component and report['label'] == label
            assert report['phase'] == 'accuracy' and report['gpu_affinity'] == '11'
            assert report['serving_batch'] == 8 and report['samples'] == report['calls'] == 0
            assert (report['isl'], report['osl'], report['tp'], report['ep']) == (4096, 16, 1, 1)
            assert len(report['results']) == case_count
            assert all(case['accuracy'] == 'PASS' for case in report['results'])
            assert report['driver_sha256'] == launch['driver_sha256']['benchmark_simple_canonical_v1.py']
            assert report['reference_driver_sha256'] == launch['driver_sha256']['benchmark_simple.py']
            for filename, expected in report['libraries'].items():
                host = Path(filename.replace('/root/', '/home/syk/', 1))
                assert digest(host) == expected
            provider_reports[label] = {
                'original_test_count': test_count, 'original_log_sha256': digest(original_log),
                'original_exit': 0, 'representative_reference_cases': case_count,
                'reference_report_sha256': digest(report_path),
                'normal_package': report['package'], 'actual_native_libraries': report['libraries'],
            }
        assert [case['case'] for case in json.loads((evidence / 'accuracy-production-before.json').read_text())['results']] == [case['case'] for case in json.loads((evidence / 'accuracy-production-after.json').read_text())['results']]
        reports[component] = {'providers': provider_reports,
                              'grant_sha256': digest(run / f'{component}-grant.json'),
                              'package_manifest_sha256': grant['package_manifest_sha256']}
    report = {'version': 'simple-normal-accuracy-v4-independent-audit', 'actual_exit': 0,
              'epoch': time.time(), 'components': reports,
              'original_tests_per_provider': 69, 'representative_reference_cases_per_provider': 14,
              'no_skips_or_tolerance_changes': True, 'no_timing_events_graphs_or_warmup': True,
              'scope': 'normal SOURCE tested-native comparator and genuine AFTER original API/representative contracts',
              'model_config_embedding8192_supplement_included': False,
              'whole_model_or_raw_trace_replay_claimed': False,
              'cross_provider_raw_output_bitwise_claimed': False,
              'GPU_work_started_by_auditor': False, 'performance_acceptance_complete': False}
    with (ROOT / 'SIMPLE_NORMAL_ICPX_ACCURACY_AUDIT_v4.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'original_tests_per_provider': 69,
                      'representative_reference_cases_per_provider': 14}))


if __name__ == '__main__':
    main()

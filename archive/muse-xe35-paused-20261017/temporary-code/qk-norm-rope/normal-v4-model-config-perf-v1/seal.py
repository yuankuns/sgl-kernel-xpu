import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
from run_perf import ARTIFACT, RUN, digest, save, verify


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    value = verify()
    audit = json.loads((RUN / 'final-evidence-audit.json').read_text())
    assert audit['actual_exit'] == 0 and audit['phase_count'] == 24 and audit['measured_event_call_count'] == 177408
    assert (RUN / 'final-evidence-audit.exit').read_text().strip() == '0'
    for name, expected in audit['raw_files'].items():
        assert digest(RUN / name) == expected['sha256']
    final = json.loads((RUN / 'analysis-final.json').read_text())
    assert final['cases']['decode-output-05-position-4100']['acceptance_status'] == 'regression_confirmed'
    assert not final['all_cases_unqualified_performance_nonregression']
    for name in ('decode-output-13-position-4108', 'legacy-representative-t8'):
        assert final['cases'][name]['acceptance_status'] == 'drift_requires_qualification'
    assert subprocess.run(['flock', '-n', '/tmp/intel-gpu-6.lock', 'true'], check=False).returncode == 0
    raw = {}
    excluded = {'evidence-seal.json', 'evidence-seal.log', 'evidence-seal.exit', 'write-set.json'}
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.name not in excluded:
            raw[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    save(RUN / 'evidence-seal.json', {'actual_exit': 0, 'CPU_seal_pid': os.getpid(),
        'independent_CPU_audit_handle_actual0': 70111, 'all_current_raw_and_docs_rehashed': raw,
        'prior_455_raw_SHA_unchanged': True, 'frozen_driver_sha256': value['driver_sha256'],
        'own_PR_DRAFT_sha256': digest(ARTIFACT / 'PR_DRAFT.md'),
        'performance_unqualified_acceptance': False, 'measured_pos4100_retest_over5_percent_not_hidden': True,
        'old_preopt3508_exit1_preserved_not_correctness_matched_timing_baseline': True,
        'GPU6_physical_lock_released': True, 'logical_GPU6_handoff_to_W4': False,
        'frozen_source_math_dispatch_no_changes': True, 'no_extra_GPU_or_compiler_launched': True,
        'eight_card_concurrency_claim': False})
    print(json.dumps({'actual_exit': 0, 'raw_and_docs_files': len(raw), 'prior_raw_455_unchanged': True,
        'physical_GPU6_lock_released': True, 'unqualified_performance_acceptance': False}, indent=2))


if __name__ == '__main__':
    main()

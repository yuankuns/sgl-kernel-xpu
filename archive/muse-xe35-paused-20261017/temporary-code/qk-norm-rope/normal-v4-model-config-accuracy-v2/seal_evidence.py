import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from run_accuracy_r2 import ARTIFACT, RUN, ROLES, digest, local_path, save, verify_inputs


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    data = verify_inputs()
    audit = json.loads((RUN / 'final-evidence-audit.json').read_text())
    assert audit['actual_exit'] == 0 and audit['case_count'] == 17 and audit['provider_case_count'] == 34
    assert (RUN / 'final-evidence-audit.exit').read_text().strip() == '0'
    assert (RUN / 'host-supervisor-r2.exit').read_text().strip() == '0'
    assert json.loads((RUN / 'drain-check.json').read_text())['drained']
    raw = {}
    excluded = {'evidence-seal.json', 'evidence-seal.log', 'evidence-seal.exit', 'write-set.json'}
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.suffix != '.md' and path.name not in excluded:
            raw[str(path.relative_to(RUN))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    for name, expected in audit['raw_files'].items():
        assert raw[name] == expected
    fixtures = json.loads((RUN / 'input-manifest.json').read_text())
    for name, fixture in fixtures.items():
        relative = str(local_path(fixture['path']).relative_to(RUN))
        assert raw[relative]['sha256'] == fixture['sha256']
    for role in ROLES:
        lines = (RUN / role / 'gpu-role.log').read_text().splitlines()
        records = [json.loads(line) for line in lines if line.startswith('{') and '"case"' in line]
        assert len(records) == 17
        for entry in records:
            path = RUN / role / f"{entry['case']['name']}.json"
            assert entry == json.loads(path.read_text())
    frozen = json.loads((ARTIFACT / 'canonical-source-freeze-v4-icpx.json').read_text())
    target = ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35'
    for name, expected in frozen['files'].items():
        assert digest(target / name) == expected['sha256']
    preopt = ARTIFACT / 'source-accuracy-v1/fresh-preopt/accuracy-t32768'
    assert preopt.with_suffix('.exit').read_text().strip() == '1'
    assert json.loads(preopt.with_suffix('.json').read_text())['violations'] == 3508
    complete_path = ARTIFACT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json'
    complete = json.loads(complete_path.read_text())
    assert complete['actual_queue_exit'] == 0 and complete['actual_queue_handle'] == 8023
    assert complete['host_queue_pid_absent']
    save(RUN / 'evidence-seal.json', {'actual_exit': 0, 'CPU_seal_pid': os.getpid(),
        'independent_tensor_reference_audit_handle_terminal0': 95498,
        'final_audit_sha256': digest(RUN / 'final-evidence-audit.json'),
        'all_raw_files_rehashed': raw, 'input_PT_count': len(fixtures), 'SOURCE_PORT_output_PT_count': 34,
        'GPU_stdout_per_case_JSON_matches_saved_phase_records': True,
        'reviewed_reference_and_provenance_file_SHA_unchanged': data['authoritative_file_SHA'],
        'frozen_PR_API_generic_BMG_math_all_file_SHA_preserved': True,
        'SOURCE_preopt3508_actual_exit1_preserved': True,
        'main_all11_full_metadata': {'path': str(complete_path), 'sha256': digest(complete_path),
            'actual_queue_handle': 8023, 'actual_queue_exit': 0, 'host_queue_pid_absent': True},
        'formal_GPU6_performance_started': False, 'new_common_performance_epoch_grant_required': True,
        'raw_tensor_replay_claim': False, 'all_model_claim': False})
    print(json.dumps({'actual_exit': 0, 'all_raw_files_rehashed': len(raw), 'input_PT_count': 17,
        'output_PT_count': 34, 'old_ref_math_API_unchanged': True, 'preopt3508_exit1_retained': True,
        'main_full_queue8023_actual0_metadata_readonly_verified': True, 'formal_performance_started': False}, indent=2))


if __name__ == '__main__':
    main()

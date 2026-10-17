import json
from contracts import ROOT, ART, BASE, WORKLOAD, code_hashes, digest, write_json


def main():
    proof = json.loads((ROOT / 'CPU_PREPARATION.json').read_text())
    assert proof['actual_exit'] == 0 and not proof['GPU_initialized']
    assert len(proof['cases']) == 51 and proof['model51_GPU_precision_not_run']
    assert (ROOT / 'CPU_FINAL_BINDING.exit').read_text().strip() == '0'
    for name in ('contracts.py', 'accuracy_driver.py', 'prepare_cpu.py', 'cpu_contract_tests.py'):
        assert proof['driver_files_sha256'][name] == digest(ROOT / name)
    template = {'issuer': 'main', 'phase': 'fmha-preopt-model51-accuracy-v1', 'gpu': 10,
                'runtime_cpus': '96-111', 'provider': 'genuine-preopt-ab3-fresh-tested-native',
                'source_head': BASE, 'workload': WORKLOAD, 'cases': 51, 'checked_elements': 808255488,
                'atol': 5e-3, 'rtol': 5e-3, 'reference_chunks': 64,
                'no_timing_warmup_events_graphs': True, 'cross_algorithm_bitwise_required': False,
                'coordinator_release': False, 'closed': False, 'runtime_reuse_permitted': False,
                'issued_epoch': None, 'epoch': None,
                'plan_sha256': digest(ART / 'model-config51-v1/plan.json'),
                'baseline_audit_sha256': digest(ART / 'v5-gpu5-performance-preparation/ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json'),
                'package_manifest_sha256': proof['provider']['package_manifest_sha256'],
                'input_code_bindings_sha256': digest(ROOT / 'input-code-bindings.json'),
                'driver_files_sha256': code_hashes(), 'CPU_gate_sha256': digest(ROOT / 'CPU_PREPARATION.json'),
                'tested_native_sha256': proof['provider']['three_tested_native_sha256'],
                'fixture_file_sha256': proof['fixture_file_sha256']}
    write_json(ROOT / 'GRANT_FIELDS_TEMPLATE_NOT_AUTHORIZED.json', template)
    record = {'CPU_actual_exit': 0, 'CPU_handle': 79973, 'CPU_supervisor_PID': 3308018,
              'CPU_container_python_PID': 3308082, 'CPU_affinity': proof['actual_container_CPU_affinity'],
              'cases': 51, 'GPU_accuracy_run': False, 'GPU_initialized': False,
              'full_reference_elements_required': 808255488,
              'driver_files_sha256': code_hashes(), 'CPU_gate_sha256': digest(ROOT / 'CPU_PREPARATION.json'),
              'first_CPU_failure_exit': 1, 'first_CPU_failure_scope': 'Overstrict lazy DSO mapping expectation; raw preserved',
              'accuracy_output_directory': 'accuracy-run-v1',
              'grant_template_sha256': digest(ROOT / 'GRANT_FIELDS_TEMPLATE_NOT_AUTHORIZED.json'),
              'baseline_compiler_scope': proof['provider']['compiler_scope']}
    write_json(ROOT / 'READY.json', record)
    entries = {path.name: digest(path) for path in sorted(ROOT.iterdir()) if path.is_file() and path.name != 'EVIDENCE_MANIFEST.json'}
    write_json(ROOT / 'EVIDENCE_MANIFEST.json', {'files_sha256': entries, 'GPU_run': False, 'CPU_actual_exit': 0})
    print(json.dumps(record))


if __name__ == '__main__':
    main()

import ast
import json
from pathlib import Path
import subprocess
import sys

import forensic_runner as forensic


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
OLD = ART / 'v5-capture-forensic-v1'


def definitions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def main():
    old_freeze = json.loads((OLD / 'CPU_READY_IMMUTABLE_FREEZE_v1.json').read_text())
    old_terminal = json.loads((ART / 'v5-capture-forensic-launch-v1/TERMINAL_SEAL_CPU_v1.json').read_text())
    for name in ('forensic_runner.py', 'pair_controller.py', 'test_cpu_contract.py'):
        assert forensic.digest(OLD / name) == old_freeze['file_SHA'][name]
    for relative, checksum in old_terminal['file_SHA256'].items():
        if relative.endswith(('.log', '.json', '.exit')):
            assert forensic.digest(ART / relative) == checksum
    for name in ('pair_controller.py', 'test_cpu_contract.py'):
        assert (OLD / name).read_bytes() == (ROOT / name).read_bytes()
    before = definitions(OLD / 'forensic_runner.py')
    after = definitions(ROOT / 'forensic_runner.py')
    assert set(after) - set(before) == {'parse_fwd_arguments'}
    assert all(before[name] == after[name] for name in before if name != 'capture_observer')
    assert forensic.CAPTURE_COUNT == 8 and forensic.GENERATIONS == 2
    assert not (ROOT / 'future-run-v1').exists()
    no_gpu = ROOT / 'never-created-by-old-grant.json'
    rejected = subprocess.run([sys.executable, '-S', str(ROOT / 'forensic_runner.py'), '--role', 'port',
        '--grant', str(ART / 'v5-capture-forensic-launch-v1/gpu5-paired-forensic-barrier-v1.json'),
        '--output', str(no_gpu)], capture_output=True, text=True)
    assert rejected.returncode == 2 and not no_gpu.exists()
    records = {}
    for role, count in (('source', 29), ('port', 33)):
        assert (ROOT / f'{role}-CPU-gate.exit').read_text().strip() == '0'
        assert (ROOT / f'{role}-registered-schema-CPU.exit').read_text().strip() == '0'
        record = json.loads((ROOT / f'{role}-registered-schema-CPU.json').read_text())
        assert record['schema_argument_count'] == count and record['actual_exit'] == 0
        assert record['CPU_native_fwd_executed'] is False and record['GPU_initialized'] is False
        assert record['all_observer_read_fields_validated']
        records[role] = record
    assert (ROOT / 'CPU-contract-v2.exit').read_text().strip() == '0'
    files = [path for path in ROOT.iterdir() if path.suffix in ('.py', '.patch', '.md', '.log', '.exit', '.json')]
    result = {
        'actual_exit': 0, 'CPU_READY': True, 'GPU_started': False, 'new_GPU_grant_issued': False,
        'old_grant_rejected_before_Torch': True, 'v1_code_and_failed_raw_log_json_exits_unchanged': True,
        'v1_full_snapshot_hashes_reused_from_terminal_seal_not_reaudited': True,
        'only_runtime_function_delta': 'extract pure parse_fwd_arguments and replace erroneous causal read with registered is_causal',
        'all_other_runtime_functions_AST_identical': True,
        'controller_and_existing_8_tests_byte_identical': True,
        'roles': {role: {'argument_count': row['schema_argument_count'], 'maps': row['normal_mapped_count'],
            'actual_CPU_backend_registered_call_captured': row['actual_CPU_backend_registered_call_captured'],
            'all_observer_read_fields_validated': row['all_observer_read_fields_validated']} for role, row in records.items()},
        'capture_count': 8, 'generations': 2, 'application_disk_upper_bound_bytes': 32 * 1024**3,
        'full_reference_evaluations_if_complete': 36,
        'checked_reference_elements_if_complete': 3623878656,
        'file_SHA': {path.name: forensic.digest(path) for path in sorted(files)},
        'v1_terminal_seal_SHA': forensic.digest(ART / 'v5-capture-forensic-launch-v1/TERMINAL_SEAL_CPU_v1.json'),
    }
    with (ROOT / 'CPU_READY_IMMUTABLE_FREEZE_v2.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('actual_exit', 'CPU_READY', 'GPU_started',
        'old_grant_rejected_before_Torch', 'all_other_runtime_functions_AST_identical', 'roles')}))


if __name__ == '__main__':
    main()

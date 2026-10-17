import json
from pathlib import Path

from forensic_runner import DIRECTORY, ROOT, digest, save


for name, expected in (('CPU-contract-r1.exit', '0'), ('CPU-audit-r1.exit', '0'), ('original-terminal-incomplete-rejected-v1.exit', '2')):
    assert (DIRECTORY / name).read_text().strip() == expected
contract = json.loads((DIRECTORY / 'CPU-PROPOSAL-CONTRACT.json').read_text())
audit = json.loads((DIRECTORY / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json').read_text())
assert contract['actual_exit'] == audit['actual_exit'] == 0 and contract['tests'] == 6
assert contract['forensic_runner_SHA256'] == digest(DIRECTORY / 'forensic_runner.py')
assert audit['completed_family_cases'] == 64 and audit['result_json_count'] == 65
assert audit['mask_failure']['actual_failed_tensor_saved'] is False
for name, record in audit['files'].items():
    assert digest(name) == record['sha256'], 'Old readonly evidence changed'
nv = ROOT / 'nvfp4-woq/normal-v5-performance-gpu1-v1'
nv_ready = json.loads((nv / 'CPU_READY.json').read_text())
for name in ('CPU-contract-r2.exit', 'CPU-provider-source-r2.exit', 'CPU-provider-after-r2.exit', 'CPU-READY-r2.exit', 'CPU-ready-consumer-r1.exit'):
    assert (nv / name).read_text().strip() == '0'
save(DIRECTORY / 'CPU_PROPOSAL_READY_SEAL.json', {'actual_exit': 0, 'contract_tests': 6, 'independent_SOURCE_readonly_audit_exit': 0, 'original_partial_run_analysis_actual_exit': 2, 'original_performance_accepted': False, 'GPU_started': False, 'compiler_started': False, 'fresh_forensic_grant_created': False, 'fresh_forensic_CPU_ready_not_GPU_acceptance': True, 'failed_tensors_plan': 'new actual whole input/mapped/invalid PT + metadata saved before exact assertions; old failed values unknown/unrecoverable from retained raw', 'storage_pointer_plan': 'capture/instantiate/replay input/mapped/invalid data_ptr/storage_ptr/offset/shape/stride and Python identities', 'native_arg_address_limit': 'keep actual native DOT hCommandList/hKernel/pArguments/arg typed labels; stock DOT omits tensor argument value addresses, no invented native binding proof', 'FM_shared_cause_claimed': False, 'forensic_runner_SHA256': digest(DIRECTORY / 'forensic_runner.py'), 'CPU_audit_SHA256': digest(DIRECTORY / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json'), 'NV1_prepared_ready_SHA256': digest(nv / 'CPU_READY.json'), 'NV1_stage': 'canonical-v5-icpx genuine normal AFTER vs fresh d983/e878 four tested-native SOURCE, legacy genuine common_ops retained', 'NV1_actual_CPU_ready_exit': nv_ready['actual_exit'], 'NV1_GPU_started': False, 'NV1_fresh_performance_grant_created': False, 'NV1_production_reference_scope': nv_ready['production_reference'], 'NV1_matrix': {'representative_original_MXFP8': nv_ready['representative_shapes_total_routed_rows'], 'production_B8_BF16': nv_ready['production_shapes'], 'rounds': 3, 'order': ['source', 'after', 'after', 'source'], 'warmup': 10, 'samples_per_case_phase': 20}, 'files': {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size} for path in DIRECTORY.iterdir() if path.is_file()}})
print(json.dumps({'actual_exit': 0, 'CPU_proposal_READY': True, 'old_partial_acceptance': False, 'NV1_CPU_READY': True, 'GPU_started': False}))

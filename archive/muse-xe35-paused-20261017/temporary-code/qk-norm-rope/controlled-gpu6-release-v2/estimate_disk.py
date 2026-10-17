import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'controlled-method-runner-cpu-v2'))
from protocol import load_plan, phase_plan, require, save

require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'Light CPU0–15 estimate')
plan, fields, cases = load_plan()
qkv_bytes = sum(case['case']['tokens'] * 18432 * 2 for case in cases)
phases = len(phase_plan(cases))
sentinel_bytes = phases * 2 * 2 * qkv_bytes
graph_outputs = phases * 2 * 2 * 2 * qkv_bytes
final_witness = 2 * 3 * 2 * qkv_bytes
failure_bytes = 2 * max(case['case']['tokens'] * 18432 * 2 for case in cases)
success = sentinel_bytes + graph_outputs + final_witness
reserve = 380 * 1024**3
save(ROOT / 'DISK_ADMISSION.json', {'actual_CPU_estimate_exit': 0, 'metadata_only_no_giant_rehash': True,
    'all19_fixture_qkv_bytes': qkv_bytes, 'phases': phases, 'graph_sets': phases * 19 * 2 * 2,
    'tensor_payload_success_bytes': success, 'tensor_payload_success_GiB': success / 1024**3,
    'sentinel_payload_GiB': sentinel_bytes / 1024**3, 'proof_and_timing_full_output_payload_GiB': graph_outputs / 1024**3,
    'final_original_PGR_witness_payload_GiB': final_witness / 1024**3,
    'latest_first_failure_tensor_payload_max_GiB': (success + failure_bytes) / 1024**3,
    'new_disk_reservation_requested_GiB': 380, 'metadata_rawtrace_zip_headroom_GiB': (reserve - success - failure_bytes) / 1024**3,
    'raw_trace_size_not_mathematically_bounded_by_tensor_shapes': True,
    'combined_QK_380_Residual_640_headroom_100_GiB': 1120,
    'current_coordinator_available_GiB_approx': 902, 'combined_reservation_shortfall_GiB_approx': 218,
    'GPU_launch_on_hold_until_coordinator_total_disk_admission': True, 'no_scope_or_evidence_reduction': True})
print(f'QK success{success/1024**3:.9f}GiB, late firstfail{(success+failure_bytes)/1024**3:.9f}GiB tensorpayload; request380GiB total reservation; hold launch', flush=True)

import json
import shutil
import time
from pathlib import Path
from prepare_cpu import ROOT, ART, WORKLOAD, digest, scope_guard


def main():
    scope_guard()
    providers = {role: json.loads((ROOT / (role + '-CPU-provider.json')).read_text()) for role in ('preopt', 'port')}
    for role, proof in providers.items():
        assert proof['actual_exit'] == 0 and not proof['GPU_initialized']
        log_version = 'v1' if role == 'preopt' else 'v2'
        assert (ROOT / (role + '-CPU-preparation-' + log_version + '.exit')).read_text().strip() == '0'
        assert len(proof['cases']) == 51
    assert (ROOT / 'CPU_CONTRACTS.exit').read_text().strip() == '0'
    for before, after in zip(providers['preopt']['cases'], providers['port']['cases'], strict=True):
        assert before['name'] == after['name'] and before['input_sha256'] == after['input_sha256']
        assert before['full_output_snapshot_bytes'] == after['full_output_snapshot_bytes']
    output_bytes = sum(case['full_output_snapshot_bytes'] for case in providers['preopt']['cases'])
    input_bytes = sum(case['full_input_snapshot_bytes'] for case in providers['preopt']['cases'])
    assert output_bytes == 808255488 * 2
    phases = {'eager_fullref': 1, 'post_eager_warmup': 1, 'first_graph_replay_all8': 8,
              'post_graph_warmup_all8': 8, 'post_timing_all8': 8}
    slots = [{'round': round_index, 'slot': slot, 'role': role} for round_index in (1, 2, 3)
             for slot, role in [('B0', 'port'), ('A1', 'preopt'), ('A2', 'preopt'), ('B3', 'port')]]
    output_snapshot_bytes = output_bytes * len(slots) * sum(phases.values())
    gib = 1024 ** 3
    evidence_cap_bytes = 512 * gib
    headroom_bytes = 100 * gib
    graph_log_budget = 8 * gib
    failure_budget = 8 * gib
    serialization_budget = 4 * gib
    planned_bytes = output_snapshot_bytes + input_bytes + graph_log_budget + failure_budget + serialization_budget
    assert planned_bytes < evidence_cap_bytes
    available = shutil.disk_usage(ROOT).free
    residual_reserve = (640 + 100) * gib
    qk_admitted_reserve = 380 * gib
    fm_forensic_reserve = 32 * gib
    candidate_margin = available - residual_reserve - evidence_cap_bytes - headroom_bytes
    paths = {'preopt51_terminal_manifest': ART / 'preopt-model51-accuracy-v2/TERMINAL_EVIDENCE_MANIFEST.json',
             'preopt51_CPU_terminal_review': ART / 'preopt-model51-accuracy-v2/CPU_TERMINAL_REVIEW.json',
             'preopt51_accuracy': ART / 'preopt-model51-accuracy-v2/accuracy-run-v1/preopt-model51-accuracy.json',
             'preopt_manifest': ART / 'matched-native-icpx-v1/preopt/normal-package-manifest.json',
             'port51_accuracy': ART / 'v5-model51-gpu10-preparation/normal-v5-gpu10-accuracy.json',
             'port_manifest': ART / 'canonical-v5-icpx/source-binary-manifest.json',
             'compiler_policy': ART / 'canonical-v5-icpx/compiler-policy-verified.json',
             'port_full_audit': ART / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json',
             'model51_plan': ART / 'model-config51-v1/plan.json',
             'original_performance_protocol': ART / 'v5-gpu5-performance-preparation/performance_driver.py',
             'original_reference': ART / 'launch-v3-icpx/source_only_accuracy.py',
             'original_backend_fixture': ART / 'launch-v3-icpx/model_semantic_accuracy.py',
             'port_graph_failure': ART / 'v5-gpu5-source-port-performance-v1/FAILED_TERMINAL.md',
             'forensic_CPU_plan': ART / 'v5-capture-forensic-v1/READY.md'}
    bindings = {name: {'path': str(path), 'sha256': digest(path)} for name, path in paths.items()}
    plan = {'phase': 'fmha-true-preopt-canonical-v5-model51-gpu3-three-BAAB', 'workload': WORKLOAD,
            'gpu': 3, 'runtime_cpus_proposed': '96-111', 'rounds': 3, 'slots': slots,
            'cases': [case['name'] for case in providers['preopt']['cases']], 'cases_per_slot': 51,
            'expected_provider_case_runs': 612, 'warmup': 5, 'graph_capture_fwd_calls': 8,
            'graph_warmup_replays': 5, 'device_event_samples_per_case': 21,
            'expected_raw_samples': 12852, 'expected_real_graphs': 612,
            'timing_unit': 'device_event_ms_per_original_backend_call = graph_replay_ms / 8',
            'graph_keep_mode': 'Original XPUGraph()/torch.xpu.graph default keep_graph=False; no unsolicited keep_graph switch',
            'fullrefs': {'atol': 5e-3, 'rtol': 5e-3, 'query_chunks': 64, 'elements_per_provider_sweep': 808255488,
                         'eager_and_first_replay_and_post_timing_actual_independent_CPU_FP32_reference': True,
                         'every_saved_stage_full_wholebytes_matches_own_reference_verified_golden': True,
                         'warmup_stage_fullref_proof': 'Complete byte equality to own independently full-reference-verified golden; do not label as another oracle call',
                         'all8_output_reference_proof': 'Each full tensor byte equality; actual independent CPU helper at graph boundary; eight separate oracle calls not claimed'},
            'snapshot_outputs_per_case': phases, 'capture_retention_change_requires_main_method_approval': True,
            'actual_graph_required': 'Real DOT/fileSHA/node graph/kernel names +8 actual registered fwd calls/maps/queue/pointers; not 8 total nodes or fakeGraph',
            'no_cross_algorithm_bitwise_preset': True, 'drop_outliers': False,
            'comparison_column': 'true-preopt-ab3_to_canonical-v5_NET_CHANGE',
            'source_port_column': 'independent optimized08955_to_PORT fidelity; not populated by this plan',
            'first_failure_stops_all_slots': True, 'performance_acceptance': False,
            'runtime_authorized': False, 'bindings': bindings}
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    disk = {'epoch': time.time(), 'CPU_only': True, 'df_available_bytes': available,
            'output_bytes_per_full51_sweep': output_bytes, 'snapshots_per_case': sum(phases.values()),
            'all12_slots_output_snapshot_bytes': output_snapshot_bytes,
            'all51_unique_complete_inputs_snapshot_bytes': input_bytes,
            'input_dedup_contract': 'One complete cloned actual input packet per case, all twelve slot SHA identical to immutable originals; fail saves actual mismatch packet; no output dropping/dedup',
            'graph_raw_log_cap_bytes': graph_log_budget, 'extra_first_failure_packet_bytes': failure_budget,
            'serialization_overhead_reserve_bytes': serialization_budget, 'calculated_evidence_bytes': planned_bytes,
            'evidence_cap_bytes': evidence_cap_bytes, 'additional_free_headroom_bytes': headroom_bytes,
            'residual_plan_reserve_bytes': residual_reserve, 'residual_currently_admitted': False,
            'QK_fullsuite_reservation_bytes': qk_admitted_reserve,
            'FM5_forensic_reservation_bytes': fm_forensic_reserve,
            'extra_failure_packet_cap_is_not_a_full_process_core_cap': True,
            'system_core_and_driver_cache_disk_policy_requires_main_admission': True,
            'current_admitted_plus_FM_request_and_headroom_bytes': qk_admitted_reserve + fm_forensic_reserve + evidence_cap_bytes + headroom_bytes,
            'remaining_after_current_admissions_and_FM_request_bytes': available - qk_admitted_reserve - fm_forensic_reserve - evidence_cap_bytes - headroom_bytes,
            'remaining_after_residual_bytes': available - residual_reserve,
            'remaining_after_both_plans_and_headroom_bytes': candidate_margin,
            'global_resource_admission': False,
            'reason': 'Coordinator now admits QK380GiB and FM forensic32GiB, residual640GiB remains held. Current admissions+requested FM512+100headroom require1024GiB, exceeding current available bytes. Residual+FM cannot coexist either. Queue/serialize after main actual resource admission; do not shrink scope or remove evidence.'}
    (ROOT / 'DISK_BUDGET_AND_ADMISSION.json').write_text(json.dumps(disk, indent=2) + '\n')
    template = {'issuer': 'main', 'phase': plan['phase'], 'gpu': 3, 'runtime_cpus': '96-111',
                'issued_epoch': None, 'epoch': None, 'coordinator_release': False, 'closed': False,
                'runtime_reuse_permitted': False, 'performance_authorized': False, 'workload': WORKLOAD,
                'plan_sha256': digest(ROOT / 'plan.json'), 'disk_budget_sha256': digest(ROOT / 'DISK_BUDGET_AND_ADMISSION.json'),
                'bindings': bindings, 'CPU_provider_sha256': {role: digest(ROOT / (role + '-CPU-provider.json')) for role in providers},
                'all_resource_admission_required': True, 'live_performance_compiler_and_maintenance_quiet_required': True,
                'capture_lifetime_and_method_review_required': True}
    (ROOT / 'GRANT_FIELDS_NOT_AUTHORIZED.json').write_text(json.dumps(template, indent=2) + '\n')
    ready = {'CPU_actual_exit': 0, 'GPU_runtime_authorized': False, 'runtime_release_ready': False,
             'scope_cases': 51, 'rounds': 3, 'full_output_oracle_elements_per_provider': 808255488,
             'same_shared_input_SHA_both_roles': True, 'all_full_golden_outputs_CPU_rehashed': True,
             'blockers': ['Main fresh GPU3 grant', 'Whole resource/disk admission including Residual/QK', 'Original FM graph139 root/lifecycle method approval'],
             'plan_sha256': digest(ROOT / 'plan.json'), 'disk_budget_sha256': digest(ROOT / 'DISK_BUDGET_AND_ADMISSION.json'),
             'new_code_sha256': {path.name: digest(path) for path in ROOT.glob('*.py')}}
    (ROOT / 'CPU_READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps({'CPU_plan_exit': 0, 'all51': True, 'role_case_runs': 612, 'snapshots_GiB': output_snapshot_bytes / gib,
                      'calculated_evidence_GiB': planned_bytes / gib, 'disk_admission': False, 'GPU_started': False}))


if __name__ == '__main__':
    main()

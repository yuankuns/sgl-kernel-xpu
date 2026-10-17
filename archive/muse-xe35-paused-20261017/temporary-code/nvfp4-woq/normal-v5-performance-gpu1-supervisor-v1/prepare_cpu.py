import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


DIRECTORY = Path(__file__).resolve().parent
NV = DIRECTORY.parent
FROZEN = NV / 'normal-v5-performance-gpu1-v1'
REP = NV / 'representative-accuracy-gpu11-v1'
specification = importlib.util.spec_from_file_location('frozen_nv_performance_policy', FROZEN / 'policy.py')
policy = importlib.util.module_from_spec(specification)
specification.loader.exec_module(policy)


def disk():
    production = []
    for experts, rows, channels, inner in policy.PRODUCTION:
        production.append({'shape': [experts, rows, channels, inner],
                           'output_fp32_bytes': rows * channels * 4,
                           'c_bf16_bytes': experts * channels * inner // 16 * 2,
                           'sf_uint8_bytes': experts * channels * inner // 32})
    representative = sum(rows * channels * 4 + experts * channels * inner // 16 * 2
                         + experts * channels * inner // 32 for experts, rows, channels, inner in policy.REPRESENTATIVE)
    baseline = sum(sum(case[key] for key in ('output_fp32_bytes', 'c_bf16_bytes', 'sf_uint8_bytes')) for case in production)
    assert baseline == int(23.50390625 * 2**30)
    bound = baseline + representative + 16 * 2**20 + 24 * 2**30 + 4 * 2**30 + 4 * 2**30
    assert bound <= 64 * 2**30
    filesystem = os.statvfs(DIRECTORY)
    available = filesystem.f_bavail * filesystem.f_frsize
    reservations = {'Residual': 640, 'FM': 32, 'W4': 0, 'diagnostics': 6, 'NV': 64, 'global_free': 100}
    return {'actual_exit': 0 if available >= sum(reservations.values()) * 2**30 else 1,
            'production_baseline': production, 'production_baseline_payload_GiB': baseline / 2**30,
            'representative_baseline_payload_bytes': representative,
            'serialization_metadata_margin_bytes': 16 * 2**20,
            'first_failure_witness_reservation_GiB': 24,
            'logs_provider_resource_samples_budget_GiB': 4,
            'temporary_and_safety_margin_GiB': 4,
            'complete_conservative_bound_bytes': bound, 'complete_conservative_bound_GiB': bound / 2**30,
            'enforced_disk_cap_GiB': 64, 'enforced_timeout_hours': 12, 'enforced_global_free_GiB': 100,
            'available_bytes': available, 'available_GiB': available / 2**30,
            'coordinator_reservations_GiB': reservations,
            'remaining_after_all_reservations_GiB': available / 2**30 - sum(reservations.values()),
            'first_failure_scope': 'Preserve frozen-v1 actual logs/exits/cache/drain. Reservation is not proof of saved failed tensors; frozen driver has no full failed tensor writer and is not modified.',
            'duplicate_round_baseline_cache': False, 'new_package_copy_bytes': 0, 'compiler_started': False}


def quiet():
    expected = policy.stack()
    samples = []
    start = time.monotonic()
    while True:
        active = policy.maintenance()
        current = policy.stack()
        samples.append({'epoch': time.time(), 'active_processes': active, 'loaded_stack': current})
        if active or current != expected:
            policy.save(DIRECTORY / 'QUIET60.json', {'actual_exit': 1, 'samples': samples,
                'continuous_quiet_seconds': time.monotonic() - start,
                'loaded_stack': {'running_kernel': expected['running_kernel'], 'loaded_modules': {'xe': expected['xe']}},
                'frozen_maintenance_filter_unchanged': True, 'no_external_processes_stopped': True})
            raise RuntimeError('Actual frozen container maintenance/stack gate blocked: ' + repr(active))
        if time.monotonic() - start >= 60:
            break
        time.sleep(2)
    policy.save(DIRECTORY / 'QUIET60.json', {'actual_exit': 0, 'samples': samples,
        'continuous_quiet_seconds': time.monotonic() - start,
        'loaded_stack': {'running_kernel': expected['running_kernel'], 'loaded_modules': {'xe': expected['xe']}},
        'frozen_maintenance_filter_unchanged': True, 'no_external_processes_stopped': True})


def final_seal():
    ready = policy.verify_ready()
    representative = json.loads((REP / 'INDEPENDENT_CPU_POSTCHECK.json').read_text())
    assert representative['actual_exit'] == 0 and representative['whole_SOURCE_input_prep_output_bits']
    assert representative['independent_CPU_FP32_elements_checked'] == 753664
    assert representative['all_known_owned_descendants_drained'] and representative['GPU11_lock_probe_exit'] == 0
    for relative, sha in representative['raw_sha256'].items():
        assert policy.digest(REP / relative) == sha
    rep_ready = json.loads((REP / 'CPU_READY.json').read_text())
    for path, sha in rep_ready['dependencies_sha256'].items():
        assert policy.digest(path) == sha
    for relative, sha in rep_ready['evidence_sha256'].items():
        assert policy.digest(REP / relative) == sha
    coverage_path = FROZEN / 'production-shape-coverage-v1/PRODUCTION_SHAPE_COVERAGE.json'
    coverage = json.loads(coverage_path.read_text())
    assert coverage['actual_exit'] == 0 and not coverage['GPU_started_by_this_audit']
    for relative, sha in coverage['old_raw_seals'].items():
        assert policy.digest(policy.ACCURACY / relative) == sha
    for label in ('source', 'after'):
        assert (DIRECTORY / f'provider-{label}.exit').read_text().strip() == '0'
        record = json.loads((DIRECTORY / f'provider-{label}.json').read_text())
        assert not record['GPU_initialized'] and record['after_manifest_sha256'] == policy.EXPECTED_MANIFEST
    assert (DIRECTORY / 'contracts.exit').read_text().strip() == '0'
    assert json.loads((DIRECTORY / 'QUIET60.json').read_text())['actual_exit'] == 0
    budget = disk()
    policy.save(DIRECTORY / 'DISK_ADMISSION.json', budget)
    assert budget['actual_exit'] == 0
    evidence = [REP / name for name in ('CPU_READY.json', 'INDEPENDENT_CPU_POSTCHECK.json', 'CURRENT_CONTAINER_DRAIN.json', 'HOST_RELEASE.json')]
    evidence += [DIRECTORY / name for name in ('provider-source.json', 'provider-after.json', 'CPU_CONTRACT.json', 'QUIET60.json', 'DISK_ADMISSION.json')]
    evidence += [coverage_path, FROZEN / 'CPU_READY.json']
    policy.save(DIRECTORY / 'FINAL_CPU_READY.json', {'actual_exit': 0, 'GPU_started': False,
        'frozen_v1_dependencies_SHA256': ready['dependencies_SHA256'],
        'external_supervisor_SHA256': {str(path): policy.digest(path) for path in DIRECTORY.glob('*.py')},
        'evidence_SHA256': {str(path): policy.digest(path) for path in evidence},
        'representative_full_precision_elements': 753664, 'complete_records_required': 96,
        'positive_raw_event_samples_required': 1920, 'scope': 'SOURCE-to-normalPORT integration fidelity, not optimization benefit'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('source', 'after', 'quiet', 'seal', 'probe'))
    args = parser.parse_args()
    policy.preflight()
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    policy.verify_ready()
    if args.action in ('source', 'after'):
        policy.save(DIRECTORY / f'provider-{args.action}.json', policy.original_provider(args.action))
    elif args.action == 'quiet':
        quiet()
    elif args.action == 'seal':
        final_seal()
    else:
        print(json.dumps({'frozen_maintenance': policy.maintenance(), 'disk': disk()}))
    print(json.dumps({'actual_exit': 0, 'action': args.action, 'GPU_started': False, 'compiler_started': False}))


if __name__ == '__main__':
    main()

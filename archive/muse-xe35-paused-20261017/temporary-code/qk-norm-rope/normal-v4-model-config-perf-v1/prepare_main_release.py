import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from run_perf import RUN, digest, verify


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    path = RUN / 'main-continue-release.json'
    assert not path.exists()
    value = verify()
    drain_path = RUN.parents[1] / 'nvfp4-woq/canonical-v5-icpx/COORDINATOR_COMPILER_DRAIN.json'
    drain = json.loads(drain_path.read_text())
    assert drain['epoch'] == 1792188818.5698872 and drain['actual_audit_exit'] == 0
    assert drain['build_actual_exit'] == 0 and drain['install_actual_exit'] == 0 and drain['post_verified']
    assert drain['canonical_process_audit_drained'] and drain['matching_live_compiler_processes'] == {}
    sys.path.insert(0, str(RUN.parents[1] / 'canonical-tools'))
    from queue_pool_icpx_v2 import verify_drained

    verify_drained()
    for pid in drain['known_NV_compiler_pids_absent']:
        assert not (Path('/proc') / str(pid)).exists()
    proof = json.loads((RUN / 'paused-suite-host-proof.json').read_text())
    pid = proof['host_suite_PID']
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    assert pid == 3130317 and fields[0] == 'T' and int(fields[19]) == 8042282
    assert not (RUN / 'phase-plan.json').exists() and not list(RUN.glob('phases/*/*.json'))
    grant_sha = hashlib.sha256((RUN / 'performance-grant.json').read_bytes()).hexdigest()
    assert grant_sha == '02518a8c8e7bcdad18fc9427b4f2912406abd13579f78138ab08b0966cba9dd2'
    freeze = json.loads((RUN / 'driver-freeze.json').read_text())
    for name, checksum in freeze['files'].items():
        assert digest(RUN / name) == checksum
    release = {'version': 'main-continue-release-v1', 'epoch': time.time(),
        'authorization': 'Explicit main CONT after NV v5 handle25442 full/install/post0 and independent compiler drain0',
        'exact_host_PID': pid, 'exact_container_PID': 234687, 'process_starttick': 8042282,
        'verified_state_before_CONT': 'T', 'zero_raw_phase_JSON': True, 'phase_plan_absent_before_CONT': True,
        'old_immutable_grant_sha256': grant_sha, 'frozen_driver_files': freeze['files'],
        'drain_evidence_path': str(drain_path), 'drain_evidence_sha256': digest(drain_path),
        'drain_evidence_epoch': drain['epoch'], 'verify_drained_actual_exit': 0,
        'canonical_manifest_sha256': value['canonical_manifest_sha256'],
        'compiler_policy_sha256': value['compiler_policy_sha256'], 'source_digest': value['source_digest'],
        'SOURCE_manifest_sha256': value['source_provider']['source_manifest_sha256'],
        'SOURCE_native_sha256': value['source_provider']['source_qk_library_sha256'], 'PORT_native_SHA': value['port_native_SHA'],
        'fixture_plan_sha256': digest(RUN / 'fixture-plan.json'), 'physical_GPU': 6, 'cpus': '32-47',
        'continue_same_PID_no_restart': True, 'no_live_runner_or_old_grant_modification': True,
        'wrapper_initial_cooling_seconds_already_completed': 5, 'additional_idle_before_CONT_seconds': 5,
        'runner_each_shape_synchronize_then_cooling_seconds': 2,
        'new_post_CONT_delay_not_injected_into_live_runner': True,
        'pause_start_epoch': json.loads((RUN / 'paused-suite-container-proof.json').read_text())['epoch'],
        'eight_card_concurrent_claim': False, 'W4_handoff': False}
    path.write_text(json.dumps(release, indent=2) + '\n')
    print(json.dumps(release, indent=2), flush=True)


if __name__ == '__main__':
    main()

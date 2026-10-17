import json
from pathlib import Path
import time

import precision_v5 as runner


def main():
    runner.scope_guard()
    root = runner.ROOT
    artifact = runner.ART
    stage = artifact / runner.VERSION
    for label in ('model51-normal-v5-CPU-gate-v1', 'model51-normal-v5-CPU-gate-v2'):
        assert (artifact / f'{label}.exit').read_text().strip() == '0'
    assert (root / 'cpu-preparation-tests-v4.exit').read_text().strip() == '0'
    assert (artifact / 'COORDINATOR_FULL_ICPX_AUDIT_v5.exit').read_text().strip() == '0'
    _, _, canonical_preflight, _, _, _, _ = runner.shared_imports()
    manifest, manifest_path, _, _ = canonical_preflight()
    assert runner.digest(manifest_path) == '0730edcb0d2f1b6b960dc40c96c9e00a3fb0acb3996053014eb6ba9e6904277f'
    assert manifest['source']['source_digest'] == runner.EXPECTED_DIGEST
    assert len(manifest['binaries']) == 735
    gate_path = root / runner.CPU_GATE_NAME
    gate = json.loads(gate_path.read_text())
    assert gate['CPU_gate'] and not gate['GPU_initialized'] and len(gate['cases']) == 51
    assert gate['real_backend_and_registered_wrapper_identity'] and len(gate['regression38_case_names']) == 38
    assert gate['harness_sha256'] == runner.digest(root / 'precision_v5.py')
    assert gate['canonical_manifest_sha256'] == runner.digest(manifest_path)
    assert gate['mapped_native_sha256'] == {name: record['sha256'] for name, record in manifest['binaries'].items()}
    assert gate['compiler_policy_sha256'] == runner.digest(stage / 'compiler-policy-verified.json')
    lock_report_path = root / 'GPU10_HOST_FLOCK_PRECHECK_v1.json'
    lock_report = json.loads(lock_report_path.read_text())
    assert lock_report['GPU'] == 10 and lock_report['origin'] == 'actual host /proc/locks'
    assert 0 <= time.time() - lock_report['epoch'] < 30
    holders = lock_report['holders']
    assert not holders
    issued = time.time()
    grant = {'phase': runner.PHASE, 'gpu': 10, 'runtime_cpus': '16-31', 'coordinator_release': True,
        'canonical_version': runner.VERSION, 'source_digest': runner.EXPECTED_DIGEST,
        'workload': runner.WORKLOAD, 'issued_epoch': issued, 'epoch': issued + 120,
        'runtime_reuse_permitted': False, 'closed': False, 'no_timing_warmup_events_graphs': True,
        'official_performance_wave': False, 'NV_handoff_authorized': False,
        'FM_GPU5_performance_lease_retained': True,
        'harness_sha256': runner.digest(root / 'precision_v5.py'),
        'plan_sha256': runner.digest(artifact / 'model-config51-v1/plan.json'),
        'source51_raw_sha256': runner.digest(artifact / 'model-config51-v1/source-accuracy.json'),
        'source38_raw_sha256': runner.digest(artifact / 'source-only-v1-fresh-optimized.json'),
        'canonical_manifest_sha256': runner.digest(manifest_path),
        'compiler_policy_sha256': runner.digest(stage / 'compiler-policy-verified.json'),
        'coordinator_full_audit_sha256': runner.digest(artifact / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json'),
        'normal_CPU_gate_sha256': runner.digest(gate_path),
        'GPU10_flock_check_epoch': lock_report['epoch'], 'GPU10_flock_check_holders': holders,
        'GPU10_flock_device_inode': lock_report['device_inode'],
        'GPU10_host_flock_precheck_sha256': runner.digest(lock_report_path),
        'source_model51_route': 'original optimized prefill6/decode45, not generic',
        'expected_current_v5_model51_route': 'true private optimized prefill6/decode45 with original backend sentinel0 resolved only in approved entry',
        'reference_scope': 'old38 plus actual model51 all-query/request/head/channel strict005 and SOURCE whole-output bits',
        'failure_policy': 'save actual precision/route failure and stop; do not launch performance or next suite',
        'authorization': 'main latest explicit conditional GPU10 precision-only authorization after current v5 CPU gates0; no GPU5 perf',
        'canonical_default_all_install_post_independent_audit_actual0': True}
    path = stage / 'gpu10-precision-only-barrier-v1.json'
    issued_path = stage / 'gpu10-precision-only-barrier-v1-issued.json'
    assert not path.exists() and not issued_path.exists()
    text = json.dumps(grant, indent=2) + '\n'
    with path.open('x') as stream:
        stream.write(text)
    with issued_path.open('x') as stream:
        stream.write(text)
    print(json.dumps({'grant_path': str(path), 'issued_epoch': issued, 'epoch': grant['epoch'],
                      'CPU_gate_sha256': grant['normal_CPU_gate_sha256'], 'GPU10_flock_holders': holders}), flush=True)


if __name__ == '__main__':
    main()

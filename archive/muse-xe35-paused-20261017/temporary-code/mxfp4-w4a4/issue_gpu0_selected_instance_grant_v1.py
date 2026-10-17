import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    directory = Path(__file__).resolve().parent / 'gpu0-selected-instance-cpu-v1'
    import sys
    sys.path.insert(0, str(directory))
    from protocol import digest, save
    ready_path = directory / 'CPU-READY.json'
    assert digest(ready_path) == 'f2a5127c97fcfb2988efb1f236d195c780a8e0e24b0db1582d1914bed0f36397'
    ready = json.loads(ready_path.read_text())
    assert ready['CPU_READY'] and not ready['GPU_started'] and not ready['GPU_grant_issued']
    assert len(ready['bound_files_SHA256']) == 895
    for name, expected in ready['bound_files_SHA256'].items():
        if not name.startswith('/opt/'):
            assert digest(name) == expected, name
    for name in ['fresh-pregrant-providers.exit', 'fresh-owned-drain-resource-CPU.exit']:
        assert int((directory / name).read_text()) == 0, name
    drain = json.loads((directory / 'fresh-owned-drain-resource-CPU.json').read_text())
    assert time.time() - drain['epoch'] < 120
    assert drain['old_W4_container_PID_absent'] == 262526 and drain['no_GPU0_prior_runtime']
    assert drain['resource_monitor_SHA256'] == digest(directory / 'resource_monitor.py')
    for kind in ('source', 'after'):
        proof = json.loads((directory / f'fresh-pregrant-{kind}.json').read_text())
        assert proof['actual_CPU_provider_exit'] == 0 and proof['XPU_initialized'] is False
        assert proof['CPU_affinity'] == list(range(16))
    owners = [3269093, 3269094, 3269943]
    assert all(not Path(f'/proc/{pid}').exists() for pid in owners)
    assert not (directory / 'host-supervisor.json').exists()
    assert not (directory / 'container-supervisor.json').exists()
    assert shutil.disk_usage(directory).free > 6 * 1024**3
    assert subprocess.run(['flock', '-n', '/tmp/intel-gpu-0.lock', 'true']).returncode == 0
    extra = {str(directory / name): digest(directory / name) for name in ['resource_monitor.py',
             'fresh-pregrant-source.json', 'fresh-pregrant-after.json', 'fresh-pregrant-providers.exit',
             'fresh-owned-drain-resource-CPU.json', 'fresh-owned-drain-resource-CPU.exit']}
    epoch = time.time()
    grant = {'issuer': 'main', 'release': True, 'grant_epoch': epoch, 'start_epoch': epoch + 120,
             'authorization': 'Latest explicit user delegates issuance after fresh gates; once GPU0 unmeasured SOURCE->normalAFTER two-original-largeM selected-instance/grid/defaultcapture1and8 diagnosis.',
             'gpu': 0, 'diagnostic_only': True, 'performance_authorized': False,
             'CPU_affinity': list(range(16)), 'OMP_NUM_THREADS': 8, 'once_only': True, 'no_automatic_retry': True,
             'first_failure_complete_output_raw_true_exit_stop_suite': True,
             'CPU_READY_SHA256': digest(ready_path), 'bound_files_SHA256': ready['bound_files_SHA256'],
             'fresh_GPU0_owned_drain_and_lockfree_verified': True, 'old_host_owners_absent': owners,
             'resource_monitor_SHA256': digest(directory / 'resource_monitor.py'),
             'supplemental_CPU_and_monitor_files_SHA256': extra,
             'worker_RSS_cap_bytes': 96 * 1024**3, 'tree_RSS_cap_bytes': 128 * 1024**3,
             'walltime_cap_seconds': 4 * 3600, 'artifact_disk_cap_bytes': 6 * 1024**3,
             'stricter_existing_inner_per_role_timeout900s_unchanged': True,
             'running_kernel': os.uname().release, 'grant_issuer_script_SHA256': digest(Path(__file__)),
             'no_events_timing_warmup_BAAB_compile_native_deps_helper_or_PR_changes': True}
    path = directory / 'main-gpu0-diagnostic-grant.json'
    with path.open('x') as stream:
        stream.write(json.dumps(grant, indent=2) + '\n')
    print(json.dumps({'grant_epoch': epoch, 'start_epoch': epoch + 120, 'grant_SHA256': digest(path),
                      'actual_issue_exit': 0, 'GPU0_fresh_lock_probe': 0, 'CPU_affinity': list(range(16)), 'GPU_started': False}))


if __name__ == '__main__':
    main()

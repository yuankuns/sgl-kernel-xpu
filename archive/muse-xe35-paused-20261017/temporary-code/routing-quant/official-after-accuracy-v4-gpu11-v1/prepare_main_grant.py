import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
ready_path = directory / 'AFTER_CPU_READY_MANIFEST.json'
assert sorted(os.sched_getaffinity(0)) == list(range(72, 80))
assert int((directory / 'CPU-ready.exit').read_text()) == 0
ready = json.loads(ready_path.read_text())
assert ready['CPU_all_gates_actual_zero'] and ready['current_native_count'] == 714
assert ready['nodeids_order_and_cases_equal'] and ready['original_and_adapted_nodeids_count'] == 1300
for name, expected in ready['bound_files_sha256'].items():
    assert digest(Path(name)) == expected, name
with Path('/tmp/intel-gpu-11.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    physical_probe_epoch = time.time()
    fcntl.flock(lock, fcntl.LOCK_UN)
grant_path = directory / 'main-normal-after-accuracy-gpu11-grant-v1.json'
assert not grant_path.exists()
epoch = time.time()
stack = {'uname_release': os.uname().release,
         'xe_version': Path('/sys/module/xe/version').read_text().strip(),
         'xe_srcversion': Path('/sys/module/xe/srcversion').read_text().strip()}
grant = {'issuer': 'main', 'authorization': 'latest explicit user reviewed CPU normal_precheck714/maps/inverse-byte adaptation/full1300 collection; allows future+120 accuracy11 after physical11free',
         'release': True, 'module': 'routing-quant', 'version': 'canonical-v4-icpx', 'gpu': 11,
         'CPU_affinity': list(range(72, 80)), 'OMP_NUM_THREADS': 8,
         'grant_epoch': epoch, 'start_epoch': epoch + 120,
         'ready_manifest_sha256': digest(ready_path), 'current_normal_manifest_sha256': ready['normal_current_manifest_sha256'],
         'six_production_shapes_authorized': True, 'independent_three_fixtures_authorized': True,
         'original_full_1300_authorized': True, 'performance_authorized': False, 'compile_authorized': False,
         'warmup_events_graphs_allowed': False, 'baseline_failure_remains_actual_exit': True,
         'previous_owner': 'main16533 terminal0 /51179 CPUaudit0 and explicit user physical11 handoff',
         'physical_GPU11_free_probe_epoch': physical_probe_epoch, 'running_kernel_and_loaded_xe': stack,
         'external_dpkg_DKMS_not_interfered': True, 'GPU6_performance_not_authorized': True}
grant_path.write_text(json.dumps(grant, indent=2) + '\n')
print(json.dumps(grant, indent=2))

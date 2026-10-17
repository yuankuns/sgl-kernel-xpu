import hashlib
import json
import subprocess
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
FORENSIC = DIRECTORY.parent


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as output:
        output.write(json.dumps(value, indent=2) + '\n')


assert (DIRECTORY / 'SOURCE_CPU_PROVIDER.exit').read_text().strip() == '0'
provider = json.loads((DIRECTORY / 'SOURCE_CPU_PROVIDER.json').read_text())
assert provider['actual_exit'] == 0 and provider['Torch_XPU_initialized'] is False
seal_path = FORENSIC / 'CPU_PROPOSAL_READY_SEAL.json'
seal = json.loads(seal_path.read_text())
assert seal['actual_exit'] == 0 and seal['contract_tests'] == 6
assert seal['forensic_runner_SHA256'] == digest(FORENSIC / 'forensic_runner.py')
assert seal['CPU_audit_SHA256'] == digest(FORENSIC / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json')
old_hosts = (3295490, 3295496, 3298682, 3298731, 3298755)
assert all(not Path('/proc', str(pid)).exists() for pid in old_hosts)
lock_status = subprocess.run(['flock', '-n', '/tmp/intel-gpu-7.lock', 'true']).returncode
assert lock_status == 0
pregrant = {'actual_exit': 0, 'epoch': time.time(), 'old_7_host_and_container_phase_PIDs_absent': list(old_hosts), 'GPU7_physical_lock_probe_exit': lock_status, 'SOURCE_CPU_provider_sha256': digest(DIRECTORY / 'SOURCE_CPU_PROVIDER.json'), 'GPU_started': False}
save(DIRECTORY / 'HOST_PRE_GRANT.json', pregrant)
issued = time.time()
grant = {'issuer': 'main-coordinator', 'authorization': 'explicit user main instruction: one GPU7 accuracy-only unmeasured isolated-mask forensic; no automatic prefix retry', 'scope': 'SOURCE-mask-singleton-unmeasured-forensic-v1', 'accuracy_authorized': True, 'performance_authorized': False, 'gpu': 7, 'label': 'before', 'context': 'isolated-mask-only', 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'CPU_affinity': list(range(80, 96)), 'issued_epoch': issued, 'start_epoch': issued + 120, 'runner_sha256': digest(FORENSIC / 'forensic_runner.py'), 'CPU_audit_sha256': digest(FORENSIC / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json'), 'CPU_seal_sha256': digest(seal_path), 'SOURCE_native_sha256': provider['SOURCE_native_sha256'], 'comparator_sha256': provider['comparator_sha256'], 'source_native_manifest_sha256': provider['source_native_manifest_sha256'], 'normal_stage_tree_sha256': provider['normal_stage_tree_SHA256'], 'normal_stage': provider['normal_stage'], 'normal_stage_files': provider['normal_stage_files'], 'actual_SOURCE_native_mappings': provider['actual_native_mappings'], 'Torch_provider': provider['Torch_provider'], 'Torch_native_mappings_SHA256': provider['Torch_native_mappings_SHA256'], 'CPU_provider_sha256': digest(DIRECTORY / 'SOURCE_CPU_PROVIDER.json'), 'host_pregrant_sha256': digest(DIRECTORY / 'HOST_PRE_GRANT.json'), 'input_recipe': provider['input_recipe'], 'events_authorized': False, 'timing_samples_authorized': 0, 'embedding_prefix_authorized': False, 'automatic_retry': False, 'performance_accepted': False, 'physical_wrapper_lock_required': True, 'initial_wrapper_cool_seconds': 5, 'shape_sync_cool_seconds': 2}
grant['launch_helpers_sha256'] = {path.name: digest(path) for path in DIRECTORY.glob('*.py')}
save(DIRECTORY / 'main-grant.json', grant)
print(json.dumps({'grant_sha256': digest(DIRECTORY / 'main-grant.json'), 'issued_epoch': issued, 'start_epoch': grant['start_epoch'], 'GPU7': 'authorized one unmeasured accuracy run only'}))

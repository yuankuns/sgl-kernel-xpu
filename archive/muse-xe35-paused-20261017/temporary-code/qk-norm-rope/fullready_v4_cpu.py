import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from canonical_check import local_path
from fresh_native_check import verify_provider
from runtime_check import audit

assert set(os.sched_getaffinity(0)) == set(range(32, 48))
assert (root / 'normal-fullready-v4-checker-r3.exit').read_text().strip() == '0'
log = (root / 'normal-fullready-v4-checker-r3.log').read_text()
decoder = json.JSONDecoder()
records = []
for offset in range(len(log)):
    if log[offset] == '{' and (offset == 0 or log[offset - 1] == '\n'):
        try:
            record, consumed = decoder.raw_decode(log[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get('canonical_version') == 'canonical-v4-icpx':
            records.append(record)
assert len(records) == 1
provenance = records[0]
manifest = local_path(provenance['canonical_manifest'])
manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
expected_manifest_sha = 'f07b42b02097669331ae3e44f071068ad5bc3e7b5a84913172b450d1f504cdc9'
assert manifest_sha == provenance['canonical_manifest_sha256'] == expected_manifest_sha
coordinator = json.loads((root / 'COORDINATOR_FULL_ICPX_AUDIT_v4.json').read_text())
assert coordinator['verified'] and coordinator['package_manifest_sha256'] == manifest_sha
assert coordinator['source_digest'] == provenance['ready_source_sha256']
assert coordinator['default_all_actual_exit'] == coordinator['install_actual_exit'] == 0
assert coordinator['source_postverified']
assert not json.loads((root / 'official-release-policy.json').read_text())['official_gpu_launch_authorized']
source = verify_provider('optimized-source')
preopt = verify_provider('preopt')
site = local_path(provenance['port_site']).resolve()
sys.path.insert(0, str(site))
os.environ['PYTHONPATH'] = str(site)
import torch

assert not torch.xpu.is_initialized()
mapped = audit(provenance, 'port', site)
assert not torch.xpu.is_initialized()
payload = json.loads(manifest.read_text())
expected_native = {str(site / entry['path']) for entry in payload['binaries'].values()}
assert set(mapped) == expected_native, 'Normal import must map the complete genuine installed native graph'
result = {
    'version': 'canonical-v4-icpx',
    'CPU_fullready_actual_exit': 0,
    'checker_handle_terminal0': 69450,
    'CPU_import_pid': os.getpid(),
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'source_digest': provenance['ready_source_sha256'],
    'canonical_manifest_sha256': manifest_sha,
    'compiler_policy_sha256': provenance['compiler_policy_proof']['compiler_policy_sha256'],
    'default_all_actual_exit': 0,
    'install_actual_exit': 0,
    'source_readonly_postverified': True,
    'supervisor_actual_exit': 0,
    'normal_package_CPU_import_verified': True,
    'complete_native_graph_mapped_and_SHA_verified': len(mapped),
    'mapped_native_libraries': mapped,
    'XPU_initialized': torch.xpu.is_initialized(),
    'fresh_source_provider': source,
    'fresh_preopt_provider': preopt,
    'official_GPU6_started': False,
    'official_accuracy_verified': False,
    'official_performance_verified': False,
    'official_runtime_package_pass': False,
    'new_main_fresh_GPU6_wave_grant_required': True,
    'GPU6_logical_lease_transfer_to_W4': False,
    'build_scope': coordinator['build_scope'],
    'earlier_checker_failures_retained': ['normal-fullready-v4-checker.exit', 'normal-fullready-v4-checker-r2.exit'],
}
(root / 'normal-fullready-v4.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({name: value for name, value in result.items() if name not in ('mapped_native_libraries', 'fresh_source_provider', 'fresh_preopt_provider')}, indent=2))

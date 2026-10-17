import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIRECTORY = 'formal-v4-perf-gpu4-longgraph-v1'
ROUND = ROOT / EVIDENCE_DIRECTORY
VERSION = 'canonical-v4-icpx'
RUN_PREFIX = VERSION + '-' + EVIDENCE_DIRECTORY.replace('formal-v4-perf-', 'formal-')
BF_DRAIN_FILE = ROOT.parent / 'bf16-gemm/formal-v4-perf-gpu4-v1/GPU-drained.json'
SCOPE = 'MoE independent longgraph1000 confirmation of conservative performance hold'
sys.path.insert(0, str(ROOT / 'official-v4-icpx-tools'))
from official_package_check import canonical_preflight, digest, runtime_path, runtime_provenance


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def driver_hashes():
    files = {path.name: digest(path) for path in sorted(ROUND.iterdir()) if path.suffix in ['.py', '.sh']}
    files['shared_official_package_check.py'] = digest(ROOT / 'official-v4-icpx-tools/official_package_check.py')
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return files, identity


def protected_source(manifest):
    freeze_path = ROOT / 'canonical-source-freeze-v4-icpx.json'
    frozen = json.loads(freeze_path.read_text())
    source = runtime_path(frozen['source'], ROOT)
    for name, expected in frozen['files'].items():
        path = source / name
        attributes = path.lstat()
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_size != expected['size'] or stat.S_IMODE(attributes.st_mode) != expected['mode'] or digest(path) != expected['sha256']:
            raise RuntimeError('Protected reviewed SOURCE file changed: ' + name)
    if frozen['files'] != manifest['source']['files']:
        raise RuntimeError('Installed SOURCE inventory differs from protected PR freeze')
    return {'verified': True, 'file_count': len(frozen['files']), 'source': str(source),
            'source_digest': frozen['source_digest'], 'freeze_sha256': digest(freeze_path), 'epoch': time.time()}


def package_gate():
    if os.environ.get('BATCHES', '8') != '8':
        raise RuntimeError('B8 only; reject before package/GPU work')
    manifest, manifest_file, site = canonical_preflight(ROOT, VERSION)
    protected = protected_source(manifest)
    audit_file = ROOT / 'COORDINATOR_FULL_ICPX_AUDIT_v4.json'
    audit = json.loads(audit_file.read_text())
    if not audit.get('verified') or audit.get('component') != 'moe-permutation' or audit.get('package_manifest_sha256') != digest(manifest_file):
        raise RuntimeError('Current coordinator genuine full/install/post audit required')
    return manifest, manifest_file, site, protected


def grant_preflight(root, grant_file, accuracy_only):
    if root != ROOT or accuracy_only or Path(grant_file).resolve() != ROUND / 'fresh-grant.json':
        raise RuntimeError('Only independent MoE GPU4 formal grant accepted')
    manifest, manifest_file, site, protected = package_gate()
    grant_path = ROUND / 'fresh-grant.json'
    grant = json.loads(grant_path.read_text())
    files, identity = driver_hashes()
    expected = {'stage': VERSION, 'scope': SCOPE, 'module': 'moe-permutation', 'gpu': 4,
                'phase': 'performance', 'issuer': 'main', 'coordinator_release': True,
                'calls': 1000, 'repeats': 41, 'warmup': 5, 'actual_native_1000_node_preflight_required': True,
                'graph_contract_sha256': digest(ROUND / 'graph_contract.py'),
                'approved_CPU_preparation_manifest_sha256': digest(ROOT / 'long-graph-confirmation-cpu-v1/CPU_PREPARATION_MANIFEST_FINAL.json'),
                'actual_compiler_drain_sha256': digest(ROUND / 'compiler-drain-pregrant.json'),
                'native_body_diagnosis_sha256': digest(ROOT / 'long-graph-confirmation-cpu-v1/CPU_NS_AND_NATIVE_BODY_DIAGNOSIS.json'),
                'SOURCE_normal_CPU_API_sha256': digest(ROOT / 'long-graph-confirmation-cpu-v1/CPU-normal-source.json'),
                'AFTER_normal_CPU_API_sha256': digest(ROOT / 'long-graph-confirmation-cpu-v1/CPU-normal-port.json'),
                'source_binary_manifest_sha256': digest(manifest_file),
                'source_digest': protected['source_digest'], 'source_freeze_sha256': protected['freeze_sha256'],
                'compiler_policy_sha256': digest(ROOT / VERSION / 'compiler-policy-verified.json'),
                'source_native_manifest_sha256': digest(ROOT / 'fresh-source-icpx-v1/source-native-binary-manifest.json'),
                'driver_sha256': identity, 'driver_files_sha256': files,
                'cpu_gate_sha256': digest(ROUND / 'cpu-normal-package-gate.json'),
                'full_queue_sha256': digest(ROOT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json'),
                'AFTER_precision_summary_sha256': digest(ROOT / 'v4-after-accuracy-prewave-v1/summary.json'),
                'SOURCE_precision_summary_sha256': digest(ROOT / 'source-native-icpx-v1-accuracy/summary.json'),
                'NV_stability_barrier_sha256': digest(ROUND / 'NV-stability-barrier.json'),
                'coordinator_compiler_drain_sha256': digest(ROOT.parent / 'nvfp4-woq/canonical-v5-icpx/COORDINATOR_COMPILER_DRAIN.json'),
                'previous_BF_drained_sha256': digest(BF_DRAIN_FILE),
                'previous_bf16_drained': True, 'minimum_BAAB_rounds': 3,
                'shared_SOURCE_inputs_sha256': digest(ROUND / 'shared-SOURCE-inputs.pt')}
    for key, value in expected.items():
        if grant.get(key) != value:
            raise RuntimeError('Fresh formal grant mismatch: ' + key)
    cpu_gate = json.loads((ROUND / 'cpu-normal-package-gate.json').read_text())
    if not cpu_gate['actual_normal_import_verified'] or cpu_gate['CPU_collection_actual_exit'] != 0:
        raise RuntimeError('Actual CPU normal gate0 required')
    if grant['epoch'] != grant['not_before_epoch'] or grant['epoch'] < grant['issued_epoch'] + 120:
        raise RuntimeError('Main authorization requires epoch +120s')
    if not cpu_gate['epoch'] < grant['issued_epoch'] or time.time() < grant['not_before_epoch']:
        raise RuntimeError('Fresh formal grant not yet eligible')
    if not manifest['checker_proofs']['post_epoch'] < grant['epoch']:
        raise RuntimeError('Grant must follow postverify')
    queue = json.loads((ROOT.parent / 'ALL_ELEVEN_FULL_ICPX_V4_RESULTS.json').read_text())
    if queue['actual_queue_exit'] != 0 or not queue['host_queue_pid_absent']:
        raise RuntimeError('All eleven full queue must be drained')
    if any(value['actual_default_all_exit'] or value['install_actual_exit'] or not value['source_postverified'] for value in queue['components'].values()):
        raise RuntimeError('All eleven full/install/post0 required')
    return manifest, manifest_file, site, {'path': str(grant_path), 'sha256': digest(grant_path), 'grant': grant}


def actual_normal_import(manifest, manifest_file, site, CPU_only):
    import torch
    import sgl_kernel

    if Path(sgl_kernel.__file__).resolve() != site / '__init__.py':
        raise RuntimeError('Real complete current v4 normal package required')
    mapped = runtime_provenance(ROOT, 'port', site, manifest, manifest_file)
    if not torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::mxfp4_small_permutation_xe35', 'XPU'):
        raise RuntimeError('Real normal package did not register the Xe35 op')
    if not any(Path(path).name == 'libsgl-ops-sycl-mxfp4_small_permutation.so' for path in mapped['loaded_libraries']):
        raise RuntimeError('Current AFTER native permutation library was not actually mapped')
    if CPU_only and torch.xpu.is_available():
        raise RuntimeError('CPU gate unexpectedly has available XPU; stop before any GPU work')
    return torch, sgl_kernel, mapped

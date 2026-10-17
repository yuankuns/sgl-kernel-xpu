import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
ACCURACY = ARTIFACT / 'normal-after-v5-icpx-accuracy-v4'
FRESH = ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017'
AFTER = ARTIFACT / 'canonical-v5-icpx'
SOURCE = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
SITES = {'source': FRESH / 'site', 'after': AFTER / 'site'}
ORDER = ('source', 'after', 'after', 'source')
PRODUCTION = [(256, rows, 8192, inner) for inner in (8192, 4096) for rows in (64, 262144)]
REPRESENTATIVE = [(2, 128, 128, 256), (2, 128, 256, 512), (4, 256, 256, 512), (2, 512, 512, 512)]
EXPECTED_MANIFEST = 'a4492cadb1020191435f184e2859bbe3ae47e7305a98b02e782562296a4e9dba'
EXPECTED_SOURCE = 'dd2adcfa688cd0df1cc154fbbd2ffa9105c53309f154b7aa3a7275cd6edd97a2'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def preflight():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise RuntimeError('B8-only before Torch/GPU: ' + name)
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            raise RuntimeError('Fixed production scope before Torch/GPU: ' + name)
    count = int(os.environ.get('GIT_CONFIG_COUNT', '0'))
    trusted = {os.environ.get(f'GIT_CONFIG_VALUE_{index}') for index in range(count) if os.environ.get(f'GIT_CONFIG_KEY_{index}') == 'safe.directory'}
    for path in (ARTIFACT.parents[1] / 'pr/muse-nvfp4-woq-xe35', ARTIFACT.parent / 'deps/sycl-tla-7b56'):
        if str(path) not in trusted:
            os.environ[f'GIT_CONFIG_KEY_{count}'] = 'safe.directory'
            os.environ[f'GIT_CONFIG_VALUE_{count}'] = str(path)
            count += 1
    os.environ['GIT_CONFIG_COUNT'] = str(count)


def stack():
    return {'running_kernel': platform.release(), 'xe': {name: Path('/sys/module/xe', name).read_text().strip() for name in ('version', 'srcversion', 'initstate')}}


def maintenance():
    active = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            name = (path / 'comm').read_text().strip()
            if fields[0] != 'Z' and re.match(r'^(icpx|icx|clang(?:\+\+)?(?:-[0-9]+)?|cc1(?:plus)?|ocloc|llvm-spirv|spirv-link|ninja|cmake|make|dkms|dpkg|apt(?:-get)?|unattended-upgr)$', name):
                active.append({'PID': int(path.name), 'name': name, 'start_ticks': int(fields[19])})
        except (OSError, ValueError, IndexError):
            continue
    return active


def dependencies():
    paths = list(DIRECTORY.glob('*.py')) + [ACCURACY / name for name in ('providers.py', 'canonical_check_v5.py')]
    paths += [SOURCE / 'benchmark/accuracy_reference.py', SOURCE / 'benchmark/bench_nvfp4_woq_group_gemm.py', SOURCE / 'tests/nvfp4_utils.py', ARTIFACT.parent / 'canonical-tools/overlay_snapshot.py']
    paths += [AFTER / 'source-binary-manifest.json', AFTER / 'compiler-policy-verified.json', AFTER / 'COORDINATOR_COMPILER_DRAIN.json', FRESH / 'source-binary-manifest.json', FRESH / 'compiler-policy-verified.json', ACCURACY / 'AFTER_ACCURACY_READY.json', ACCURACY / 'independent-completed-CPU-audit-v1.json']
    paths += [ACCURACY / name for name in ('supervisor.py', 'process_tree.py')]
    paths += [ARTIFACT / 'fresh-source-icpx-v2' / name for name in ('build_source.py', 'sidecar.py', 'process_tree.py')]
    paths += [ARTIFACT / 'verify_source_baseline.py', ARTIFACT / 'canonical-source-freeze-v5-icpx.json']
    return {str(path): digest(path) for path in paths}


def verify_ready():
    preflight()
    ready = json.loads((DIRECTORY / 'CPU_READY.json').read_text())
    assert ready['actual_exit'] == 0 and ready['GPU_started'] is False
    assert ready['manifest_sha256'] == EXPECTED_MANIFEST and ready['source_digest'] == EXPECTED_SOURCE
    assert ready['dependencies_SHA256'] == dependencies(), 'Prepared code/source/provider changed'
    for label in ('source', 'after'):
        path = DIRECTORY / ready['provider_files'][label]
        assert digest(path) == ready['provider_SHA256'][label]
    return ready


def verify_grant(path):
    ready = verify_ready()
    grant = json.loads(Path(path).read_text())
    expected = {'issuer': 'main-coordinator', 'scope': 'nvfp4-normal-v5-performance-gpu1-v1', 'gpu': 1, 'performance_authorized': True, 'coordinator_release': True, 'canonical_version': 'canonical-v5-icpx', 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'rounds': 3, 'order': list(ORDER), 'warmup': 10, 'samples': 20, 'CPU_affinity': list(range(80, 96)), 'manifest_sha256': EXPECTED_MANIFEST, 'source_digest': EXPECTED_SOURCE, 'CPU_READY_sha256': digest(DIRECTORY / 'CPU_READY.json')}
    assert all(type(grant.get(key)) is type(value) and grant[key] == value for key, value in expected.items()), 'Wrong/old grant'
    assert grant['dependencies_SHA256'] == ready['dependencies_SHA256']
    assert all(type(grant[key]) in (int, float) and math.isfinite(grant[key]) for key in ('start_epoch', 'issued_epoch'))
    assert grant['start_epoch'] - grant['issued_epoch'] >= 120 and grant['start_epoch'] > 0
    quiet_path = Path(grant['quiet_evidence'])
    assert digest(quiet_path) == grant['quiet_evidence_sha256']
    quiet = json.loads(quiet_path.read_text())
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    assert quiet['loaded_stack']['running_kernel'] == stack()['running_kernel']
    assert quiet['loaded_stack']['loaded_modules']['xe'] == stack()['xe']
    assert grant['running_stack'] == stack()
    assert 0 <= grant['issued_epoch'] - quiet['samples'][-1]['epoch'] <= 600
    return grant


def runtime_grant():
    preflight()
    path = os.environ.get('NV_V5_PERFORMANCE_GPU1_GRANT')
    if not path:
        raise RuntimeError('Fresh main GPU1 performance grant required before Torch/GPU')
    grant = verify_grant(Path(path))
    assert os.environ.get('ZE_AFFINITY_MASK') == '1' and time.time() >= grant['start_epoch']
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    assert not maintenance(), 'Maintenance/compiler active: no timing'
    return grant


def original_provider(label):
    sys.path.insert(0, str(ACCURACY))
    from providers import verify_provider
    return verify_provider(label, cpu_only=True)


def original_provenance():
    sys.path.insert(0, str(ACCURACY))
    from supervisor import provenance
    return provenance()

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
OLD = ARTIFACT / 'normal-after-v5-icpx-accuracy-v4'
SOURCE = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
SITES = {'source': ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017/site',
         'after': ARTIFACT / 'canonical-v5-icpx/site'}
SHAPES = ((2, 128, 128, 256), (2, 128, 256, 512),
          (4, 256, 256, 512), (2, 512, 512, 512))
INPUT_NAMES = ('a_q', 'a_s', 'b_fp4', 'b_sf', 'b_gs', 'b_mx', 'b_smx', 'problems', 'offsets')
GIB = 1024 ** 3
DISK_CAP = GIB + 256 * 1024 ** 2
DISK_HEADROOM = GIB


def preflight():
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE',
                 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            raise RuntimeError('B8-only before Torch/GPU: ' + name)
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            raise RuntimeError('Serving scope rejected before Torch/GPU: ' + name)
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    specification = importlib.util.spec_from_file_location('readonly_preparation_policy', ARTIFACT / 'normal-v5-performance-gpu1-v1/policy.py')
    policy = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(policy)
    policy.preflight()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


def provider(label):
    sys.path.insert(0, str(OLD))
    from providers import verify_provider
    return verify_provider(label, cpu_only=True)


def dependencies():
    paths = list(DIRECTORY.glob('*.py'))
    paths += [SOURCE / 'benchmark' / name for name in ('accuracy_reference.py', 'bench_nvfp4_woq_group_gemm.py')]
    paths += [SOURCE / 'tests/nvfp4_utils.py', OLD / 'providers.py', OLD / 'canonical_check_v5.py', OLD / 'process_tree.py']
    paths += [ARTIFACT / 'normal-v5-performance-gpu1-v1/policy.py', ARTIFACT / 'normal-v5-performance-gpu1-v1/CPU_READY.json',
              ARTIFACT / 'canonical-source-freeze-v5-icpx.json',
              ARTIFACT / 'canonical-v5-icpx/source-binary-manifest.json',
              ARTIFACT / 'canonical-v5-icpx/compiler-policy-verified.json',
              ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017/source-binary-manifest.json',
              ARTIFACT / 'fresh-source-icpx-v2/runs/nv-source-icpx-native-v2-20261017/compiler-policy-verified.json']
    return {str(path): digest(path) for path in paths}


def bits(tensor):
    import torch
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def evidence(tensor):
    raw = bits(tensor)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


def verify_ready():
    ready = json.loads((DIRECTORY / 'CPU_READY.json').read_text())
    assert ready['actual_exit'] == 0 and ready['GPU_initialized'] is False
    assert ready['dependencies_sha256'] == dependencies()
    for relative, sha in ready['evidence_sha256'].items():
        assert digest(DIRECTORY / relative) == sha
    return ready


def grant(path):
    ready = verify_ready()
    result = json.loads(Path(path).read_text())
    for name, expected in {'issuer': 'main-coordinator-delegated', 'scope': DIRECTORY.name,
                           'gpu': 11, 'accuracy_only': True, 'performance_authorized': False,
                           'canonical_version': 'canonical-v5-icpx', 'serving_batch': 8,
                           'ISL': 4096, 'OSL': 16, 'TP': 1, 'EP': 1,
                           'CPU_affinity': list(range(80, 96)), 'process_cap_GiB': 16,
                           'tree_cap_GiB': 24, 'minimum_available_GiB': 96,
                           'disk_cap_bytes': DISK_CAP, 'disk_headroom_bytes': DISK_HEADROOM}.items():
        assert result[name] == expected, name
    assert result['ready_sha256'] == digest(DIRECTORY / 'CPU_READY.json')
    assert result['dependencies_sha256'] == ready['dependencies_sha256']
    assert result['start_epoch'] == result['issued_epoch'] + 120
    return result

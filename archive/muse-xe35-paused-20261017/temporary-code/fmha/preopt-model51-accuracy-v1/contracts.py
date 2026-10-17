import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
BASE = 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, record):
    if path.parent.resolve() != ROOT.resolve() and ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError('Only new sidecar directory writable')
    path.write_text(json.dumps(record, indent=2) + '\n')


def scope_guard():
    for name, expected in {'SERVING_BATCH': '8', 'BATCHES': '8', 'BATCH_SIZE': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1', 'DTYPE': 'BF16'}.items():
        if os.environ.get(name, expected).strip() != expected:
            raise RuntimeError('Workload override rejected before Torch/GPU: ' + name)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    parser.add_argument('--dtype', choices=['BF16'], default='BF16')
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--grant', type=Path)
    result = parser.parse_args()
    scope_guard()
    return result


def dependencies():
    scope_guard()
    bindings = json.loads((ROOT / 'input-code-bindings.json').read_text())
    for relative, expected in bindings.items():
        if digest(ART / relative) != expected:
            raise RuntimeError('Pinned input/helper code changed: ' + relative)
    sys.path.insert(0, str(ART / 'launch-v3-icpx'))
    import model_semantic_accuracy as semantic
    import source_only_accuracy as reference
    import matched_baseline_provenance as baseline
    import check_preopt
    assert semantic.Path is Path and baseline.Path is Path and reference.Path is Path
    plan = semantic.verify_plan()
    assert plan['workload'] == WORKLOAD and plan['atol'] == plan['rtol'] == 5e-3
    assert len(plan['cases']) == 51 and plan['expected_checked_elements_per_provider'] == 808255488
    return semantic, reference, baseline, check_preopt, plan


def provider():
    semantic, reference, baseline, check_preopt, plan = dependencies()
    package, hashes = baseline.package_preflight('preopt')
    provenance = baseline.provenance('preopt')
    assert provenance['source_head'] == BASE and len(hashes) == 216
    assert package == ART / 'matched-native-icpx-v1/preopt/site/sgl_kernel'
    audit = json.loads((ART / 'v5-gpu5-performance-preparation/ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json').read_text())['preopt']
    assert audit['genuine_three_tested_native_icpx_current_SHA_verified'] and not audit['complete_original_package_all_icpx']
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    normal_import = json.loads((ART / 'matched-native-icpx-v1/preopt/normal-import-verified.json').read_text())
    required_mapped = {Path(name).name for name in normal_import['loaded_normal_package_libraries_sha256']}
    check_preopt.verify_loaded(package, hashes, required_mapped | check_preopt.REQUIRED)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    framework = semantic.import_framework(torch, os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu')
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    bundles = {name: torch.load(semantic.ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
    for name, record in plan['fixture_files'].items():
        for tensor_name, expected in record['tensors'].items():
            tensor = bundles[name][tensor_name]
            assert list(tensor.shape) == expected['shape'] and str(tensor.dtype) == expected['dtype']
            assert reference.tensor_digest(tensor, torch) == expected['sha256']
    maps = Path('/proc/self/maps').read_text()
    paths = sorted({line.split()[-1] for line in maps.splitlines() if line.split()[-1].startswith('/')})
    mapped = {name: digest(Path(name)) for name in paths if Path(name).is_file()}
    proof = {'source_head': BASE, 'normal_package': str(package), 'package_files': len(hashes),
             'package_manifest_sha256': digest(ART / 'matched-native-icpx-v1/preopt/normal-package-manifest.json'),
             'mapped_files_sha256': mapped, 'three_tested_native_sha256': {name: hashes[name] for name in sorted(check_preopt.REQUIRED)},
             'registered_schema': str(torch.ops.sgl_kernel.fwd.default._schema),
             'real_backend_and_registered_wrapper_identity': True, 'compiler_provenance': provenance,
             'compiler_scope': 'Three genuine original ab3/e878 tested math natives fresh icpx; legacy common_ops registration and non-targets historical GNU/icx; not full baseline all-icpx/default-all',
             '38_precision_not_substitute_for_model51': True}
    return torch, semantic, reference, framework, plan, bundles, proof


def observe_registered(torch, records, stop_before_native=False):
    from torch.utils._python_dispatch import TorchDispatchMode

    class RegisteredBoundaryStop(RuntimeError):
        pass

    class ObserveRegistered(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            keywords = {} if kwargs is None else kwargs
            if function != torch.ops.sgl_kernel.fwd.default:
                return function(*args, **keywords)
            parameters = {item.name: item.default_value for item in function._schema.arguments}
            parameters.update(dict(zip((item.name for item in function._schema.arguments), args)))
            parameters.update(keywords)
            actual = {}
            for name, value in parameters.items():
                if isinstance(value, torch.Tensor):
                    actual[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
                else:
                    actual[name] = value
            query, key, table = parameters['q'], parameters['k'], parameters['page_table']
            assert query.dtype == key.dtype == torch.bfloat16 and query.shape[1:] == (128, 64)
            assert key.shape[1:] == (64, 16, 64) and table.dim() == 2
            phase = 'decode' if parameters['max_seqlen_q'] == 1 else 'prefill'
            assert parameters['max_seqlen_k'] == table.shape[1] * 64
            record = {'operator': str(function), 'actual_registered_parameters': actual,
                      'phase': phase, 'original_math_binding': 'libsgl-ops-sycl-flash_attentionXe35_' + phase + '_64_bf16.so',
                      'binding_boundary': 'Original registered root dispatch/source and actual native mapped SHA, not hardware profiler trace',
                      'CPU_pre_dispatch_only': stop_before_native, 'native_operator_completed': False}
            records.append(record)
            if stop_before_native:
                raise RegisteredBoundaryStop('CPU stops at real registered op before any native work; no fake output')
            result = function(*args, **keywords)
            record['native_operator_completed'] = True
            return result

    return ObserveRegistered(), RegisteredBoundaryStop


def code_hashes():
    return {path.name: digest(path) for path in sorted(ROOT.glob('*.py'))}


def grant_guard(path):
    scope_guard()
    if path is None or not path.is_file():
        raise RuntimeError('New explicit main GPU10 accuracy grant required before Torch import; no runtime authorized')
    grant = json.loads(path.read_text())
    cpu_proof = json.loads((ROOT / 'CPU_PREPARATION.json').read_text())
    expected = {'issuer': 'main', 'phase': 'fmha-preopt-model51-accuracy-v1', 'gpu': 10,
                'runtime_cpus': '96-111', 'provider': 'genuine-preopt-ab3-fresh-tested-native',
                'source_head': BASE, 'workload': WORKLOAD, 'cases': 51, 'checked_elements': 808255488,
                'atol': 5e-3, 'rtol': 5e-3, 'reference_chunks': 64, 'no_timing_warmup_events_graphs': True,
                'cross_algorithm_bitwise_required': False, 'coordinator_release': True,
                'closed': False, 'runtime_reuse_permitted': False,
                'plan_sha256': digest(ART / 'model-config51-v1/plan.json'),
                'baseline_audit_sha256': digest(ART / 'v5-gpu5-performance-preparation/ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json'),
                'package_manifest_sha256': digest(ART / 'matched-native-icpx-v1/preopt/normal-package-manifest.json'),
                'input_code_bindings_sha256': digest(ROOT / 'input-code-bindings.json'), 'driver_files_sha256': code_hashes(),
                'CPU_gate_sha256': digest(ROOT / 'CPU_PREPARATION.json'),
                'tested_native_sha256': cpu_proof['provider']['three_tested_native_sha256'],
                'fixture_file_sha256': cpu_proof['fixture_file_sha256']}
    for name, value in expected.items():
        if grant.get(name) != value:
            raise RuntimeError('Fresh accuracy grant mismatch: ' + name)
    if grant['epoch'] < grant['issued_epoch'] + 120 or time.time() < grant['epoch']:
        raise RuntimeError('Fresh main future epoch+120 not eligible')
    if os.environ.get('FMHA_PREOPT_MODEL51_MAIN_GRANTED') != '1' or os.environ.get('ZE_AFFINITY_MASK') != '10':
        raise RuntimeError('Explicit fresh main GPU10 lease required')
    if os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('Actual runtime CPU96–111 required')
    return grant

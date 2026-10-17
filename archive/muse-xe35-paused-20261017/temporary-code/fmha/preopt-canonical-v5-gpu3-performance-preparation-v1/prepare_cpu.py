import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'dtype': 'BF16'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def scope_guard():
    for name, expected in {'SERVING_BATCH': '8', 'BATCHES': '8', 'BATCH_SIZE': '8', 'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1', 'DTYPE': 'BF16'}.items():
        if os.environ.get(name, expected).strip() != expected:
            raise RuntimeError('Workload override rejected before Torch: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--role', choices=('preopt', 'port'))
    parser.add_argument('--gpu-id', type=int, choices=[3], default=3)
    arguments = parser.parse_args()
    scope_guard()
    if arguments.guard_only:
        print('B8 GPU3 future plan guard0; no Torch/GPU import')
        return
    if arguments.role is None:
        raise RuntimeError('CPU provider role required; no runtime mode or grant issuer')
    assert os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu'
    assert os.sched_getaffinity(0) == set(range(96, 112))
    sys.path.insert(0, str(ART / 'preopt-model51-accuracy-v2'))
    import contracts as preopt
    sys.path.insert(0, str(ART / 'v5-model51-gpu10-preparation'))
    import precision_v5 as precision
    original, guards, canonical_preflight, local_path, verify_loaded, full_reference, tensor_digest = precision.shared_imports()
    if arguments.role == 'preopt':
        torch, original, reference, framework, plan, bundles, proof = preopt.provider()
        package = Path(proof['normal_package'])
        raw_path = ART / 'preopt-model51-accuracy-v2/accuracy-run-v1/preopt-model51-accuracy.json'
        assert (ART / 'preopt-model51-accuracy-v2/ACCURACY_SUPERVISOR.exit').read_text().strip() == '0'
        assert (ART / 'preopt-model51-accuracy-v2/CPU_TERMINAL_REVIEW.exit').read_text().strip() == '0'
        compiler_scope = proof['compiler_scope']
        binding = proof['three_tested_native_sha256']
        provider_maps = proof['mapped_files_sha256']
    else:
        manifest, manifest_path, package, required = canonical_preflight()
        assert manifest['source']['source_digest'] == precision.EXPECTED_DIGEST
        audit = json.loads((ART / 'COORDINATOR_FULL_ICPX_AUDIT_v5.json').read_text())
        assert audit['actual_exit'] == audit['default_all_actual_exit'] == audit['install_actual_exit'] == 0
        assert audit['package_manifest_sha256'] == digest(manifest_path)
        policy = json.loads((ART / 'canonical-v5-icpx/compiler-policy-verified.json').read_text())
        assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command']
        assert all(Path(policy[name]).name == 'icpx' for name in ('cxx', 'sycl', 'host'))
        assert Path(policy['c']).name == 'icx'
        import torch
        import sgl_kernel
        assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
        hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
        verify_loaded(package, hashes, set(hashes))
        binding = precision.verify_native_binding(package)
        assert len(precision.mapped_names(package, hashes)) == 735
        framework = original.import_framework(torch, True)
        plan = original.verify_plan()
        bundles = {name: torch.load(original.ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
        raw_path = ART / 'v5-model51-gpu10-preparation/normal-v5-gpu10-accuracy.json'
        compiler_scope = 'Genuine canonical-v5 default-all/current install/IntelLLVM CXX,SYCL,host icpx/C icx; public7b56/profilingOFF; not legacy SOURCE normal registration'
        maps = Path('/proc/self/maps').read_text().splitlines()
        paths = sorted({line.split()[-1] for line in maps if line.split()[-1].startswith('/')})
        provider_maps = {name: digest(Path(name)) for name in paths if Path(name).is_file()}
    assert not torch.xpu.is_initialized()
    assert plan['workload'] == WORKLOAD and len(plan['cases']) == 51
    assert plan['atol'] == plan['rtol'] == 5e-3
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    raw = json.loads(raw_path.read_text())
    raw_records = {record['name']: record for record in raw['cases']}
    from torch.utils._python_dispatch import TorchDispatchMode

    class CPUBoundaryStop(RuntimeError):
        pass

    rows = []

    class ObserveBoundary(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            keywords = {} if kwargs is None else kwargs
            if function != torch.ops.sgl_kernel.fwd.default:
                return function(*args, **keywords)
            actual = {item.name: item.default_value for item in function._schema.arguments}
            actual.update(dict(zip((item.name for item in function._schema.arguments), args)))
            actual.update(keywords)
            rows[-1]['actual_registered_parameters'] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)} if isinstance(value, torch.Tensor) else value for name, value in actual.items()}
            raise CPUBoundaryStop('CPU before genuine native dispatch; no fake output')

    for case in plan['cases']:
        tensors = original.fixture(case, bundles[case['bundle']], torch, 'cpu')
        golden = raw_records[case['name']]
        assert golden['reference_PASS'] and golden['checked_elements'] == math.prod(case['output_shape'])
        assert [tensor_digest(tensor, torch) for tensor in tensors] == golden['input_sha256']
        if arguments.role == 'preopt':
            output_path = ART / 'preopt-model51-accuracy-v2' / golden['output_file']
            output_sha = golden['output_tensor_sha256']
        else:
            output_path = ART / 'v5-model51-gpu10-preparation/normal-v5-full-outputs' / (golden['family'] + '-' + case['name'] + '.pt')
            output_sha = golden['output_sha256']
            assert len(golden['actual_fwd_observations']) == 1
            assert golden['actual_fwd_observations'][0]['guard_evaluation_from_actual_metadata']['optimized']
        assert digest(output_path) == golden['saved_full_output_file_sha256']
        output = torch.load(output_path, map_location='cpu', weights_only=True)
        assert output.dtype == torch.bfloat16 and list(output.shape) == case['output_shape']
        assert tensor_digest(output, torch) == output_sha
        row = {'name': case['name'], 'input_sha256': golden['input_sha256'], 'golden_output_sha256': output_sha,
               'golden_file_sha256': digest(output_path), 'golden_full_reference_elements': golden['checked_elements'],
               'full_input_snapshot_bytes': sum(tensor.numel() * tensor.element_size() for tensor in tensors),
               'full_output_snapshot_bytes': output.numel() * output.element_size()}
        rows.append(row)
        try:
            with ObserveBoundary():
                original.backend_call(case, tensors, torch, framework)
        except CPUBoundaryStop:
            pass
        else:
            raise RuntimeError('Expected registered CPU boundary was not observed')
        expected_maxK = case['source_wrapper_inferred_max_seqlen_k'] if arguments.role == 'preopt' else 0
        assert row['actual_registered_parameters']['max_seqlen_k'] == expected_maxK
        del tensors, output
    assert not torch.xpu.is_initialized() and len(rows) == 51
    assert sum(row['golden_full_reference_elements'] for row in rows) == 808255488
    record = {'actual_exit': 0, 'role': arguments.role, 'workload': WORKLOAD, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'normal_package': str(package), 'maps_file_sha256': provider_maps, 'native_binding': binding,
              'compiler_scope': compiler_scope, 'actual_registered_schema': str(torch.ops.sgl_kernel.fwd.default._schema),
              'wrapper_signature': str(inspect.signature(flash_attn_with_kvcache)), 'golden_raw_sha256': digest(raw_path),
              'plan_sha256': digest(ART / 'model-config51-v1/plan.json'), 'cases': rows,
              'GPU_initialized': False, 'GPU_native_calls': 0, 'GPU_graph_or_event_created': False,
              'CPU_fullrefs_not_rerun': True, 'prior_independent_full_reference_and_whole_saved_output_rehash': True}
    (ROOT / (arguments.role + '-CPU-provider.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'role': arguments.role, 'CPU_actual_exit': 0, 'cases': 51, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

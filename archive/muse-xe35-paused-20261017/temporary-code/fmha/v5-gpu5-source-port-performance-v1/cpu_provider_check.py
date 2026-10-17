import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
sys.path.insert(0, str(ART / 'v5-gpu5-performance-preparation'))
import performance_driver as driver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'port'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    driver.precision.scope_guard()
    original, _, canonical_preflight, _, verify_loaded, _, _ = driver.precision.shared_imports()
    manifest, manifest_path, package, required = canonical_preflight()
    if args.role == 'port':
        hashes = {name: row['sha256'] for name, row in manifest['binaries'].items()}
        required = set(hashes)
        compiler = {'whole_package_icpx': True,
            'compiler_policy_sha256': driver.precision.digest(ART / 'canonical-v5-icpx/compiler-policy-verified.json')}
        binding = driver.precision.verify_native_binding(package)
    else:
        from matched_baseline_provenance import package_preflight, provenance
        package, hashes = package_preflight('source')
        compiler = provenance('source')
        imported = json.loads((ART / 'matched-native-icpx-v1/source/normal-import-verified.json').read_text())
        required = {Path(name).name for name in imported['loaded_normal_package_libraries_sha256']}
        binding = compiler['native_kernels']
    import torch
    import sgl_kernel
    assert not torch.xpu.is_initialized()
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    verify_loaded(package, hashes, required)
    assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::fwd', 'XPU')
    framework = original.import_framework(torch, True)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    plan = original.verify_plan()
    assert len(plan['cases']) == 51
    assert not torch.xpu.is_initialized()
    verify_loaded(package, hashes, required)
    report = {'actual_exit': 0, 'role': args.role, 'GPU_initialized': False,
        'normal_package': str(package), 'mapped_native_count': len(required),
        'mapped_SHA': {name: hashes[name] for name in sorted(required)},
        'canonical_manifest_sha256': driver.precision.digest(manifest_path),
        'native_binding': binding, 'compiler_scope_and_confound': compiler,
        'model51_plan_sha256': driver.precision.digest(original.ROOT / 'plan.json'),
        'backend_wrapper_actual_identity': True, 'actual_registered_schema': str(torch.ops.sgl_kernel.fwd.default._schema),
        'driver_sha256': driver.precision.digest(ART / 'v5-gpu5-performance-preparation/performance_driver.py')}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'role': args.role, 'CPU_provider_check': 0, 'normal_mapped_SHA': len(required), 'GPU_initialized': False}))


if __name__ == '__main__':
    main()

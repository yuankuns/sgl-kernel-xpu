import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import sgl_kernel
import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    artifact = Path(__file__).resolve().parent
    stage = artifact / 'fresh-current-source-icpx-native-v3'
    preparation = artifact / 'fresh-current-source-native-v3-preparation'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert os.environ['LD_LIBRARY_PATH'].startswith('/opt/gfx-deps/profiling_tools/lib:')
    assert Path(sgl_kernel.__file__).resolve() == stage / 'site/sgl_kernel/__init__.py'
    assert str(Path(torch.__file__).resolve()).startswith('/opt/gfx-deps/venv/')
    assert not torch.xpu.is_initialized()
    required = ['configure', 'native-build', 'site-package-install', 'site-package-import', 'isolated-site-stage', 'container', 'wrapper', 'supervisor']
    assert all(int((stage / f'{name}.exit').read_text()) == 0 for name in required)
    assert all(json.loads((stage / 'source-postverify.json').read_text()).values())
    policy = json.loads((stage / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM' and len(policy['targets']) == 4
    assert policy['compiler_binary_sha256'] == digest(Path(policy['CXX']))
    targets = ['TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout', 'TripleOps']
    expected = {'sgl_kernel/libsgl-ops-sycl-' + name + '.so' for name in targets}
    manifest = json.loads((stage / 'source-native-package-manifest.json').read_text())
    assert set(manifest['changed_package_files']) == expected
    for name, fingerprint in manifest['current_install_files'].items():
        assert digest(stage / 'site' / name) == fingerprint, name
        if name not in expected:
            assert fingerprint == manifest['original_package_files'][name]
    for name, fingerprint in manifest['mapped_native_sha256'].items():
        assert digest(Path(name)) == fingerprint
    actual_maps = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if '/sgl_kernel/' in line and '.so' in line})
    assert set(actual_maps) == set(manifest['mapped_native_sha256'])
    actual_native_SHA256 = {name: digest(Path(name)) for name in actual_maps}
    old_stage = artifact / 'fresh-source-icpx-native-v2'
    old = json.loads((preparation / 'old-stage-protected-SHA256.json').read_text())
    assert {str(path.relative_to(old_stage)): digest(path) for path in old_stage.rglob('*') if path.is_file()} == old
    before = json.loads((stage / 'source-preverify.json').read_text())
    source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    dependency = Path('/root/worktrees/cutlass_internal/muse-w4a4-e87870b')
    for root, inventory in [(source, before['original_tracked']), (dependency, before['dependency_tracked'])]:
        for name, fingerprint in inventory.items():
            assert digest(root / name) == fingerprint, name
    helper = json.loads((stage / 'fresh-helper-type-verification.json').read_text())
    assert helper['BF16_silu_actual_accumulator_type'] == 'float'
    assert helper['TripleOps_SHA256'] != 'e5946388c78064722885345fec961ad6bb12caa846c369eee1beccc35d19aed6'
    resource = json.loads((stage / 'native-build.resource-summary.json').read_text())
    assert resource['drained'] and not resource['guard'] and resource['actual_exit'] == 0
    assert resource['peak_tree_rss_bytes'] <= 16 * 1024**3 and resource['peak_file_subtree_rss_bytes'] <= 8 * 1024**3
    header_paths = [Path(torch.__file__).parent / 'include/ATen/OpMathType.h',
                    Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/bfloat16.hpp'),
                    source / 'src/sycl/SYCLHelpers.h', source / 'src/sycl/MemoryAccess.h']
    assert not torch.xpu.is_initialized()
    report = {'epoch': time.time(), 'CPU_acceptance_actual_exit': 0, 'original_SOURCE_ref': before['source_HEAD'],
              'original_dependency_ref': before['dependency_HEAD'], 'normal_SOURCE_site': str(stage / 'site'),
              'normal_PACKAGE_init': sgl_kernel.__file__, 'Torch_provider': torch.__file__,
              'Torch_provider_SHA256': digest(Path(torch.__file__)), 'Torch_version': torch.__version__,
              'current_mapped_native_SHA256': actual_native_SHA256, 'mapped_native_count': len(actual_maps),
              'four_tested_native_SHA256': {name: digest(stage / 'site/sgl_kernel' / ('libsgl-ops-sycl-' + name + '.so')) for name in targets},
              'fresh_TripleOps_BF16_accumulator_actual_type': 'float', 'original_registration_Python_other_native_unchanged': True,
              'old_stage_1565_file_inventory_unchanged': True, 'protected_current_source_and_dependency_full_inventory_unchanged': True,
              'source_TripleOps_SHA256': digest(source / 'src/sycl/TripleOps.cpp'),
              'actual_current_headers_SHA256': {str(path): digest(path) for path in header_paths},
              'actual_CXX_SYCL_host': policy['CXX'], 'actual_C': policy['C'], 'compiler_SHA256': policy['compiler_binary_sha256'],
              'compiler_policy_and_ABI_and_current_package_hashes_verified': True,
              'native_build_resources': resource, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'GPU_initialized': False, 'GPU_precision_14_plus_1300_not_run_or_authorized': True,
              'not_full_all_icpx_SOURCE_package': True}
    (stage / 'cpu-native-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'current_mapped_native_SHA256'}, indent=2))


if __name__ == '__main__':
    main()

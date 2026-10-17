import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / 'routing-quant/fresh-current-source-icpx-native-v3'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    for filename in ('configure', 'native-build', 'site-package-install', 'site-package-import',
                     'isolated-site-stage', 'torch-cpu-import-preflight', 'container', 'wrapper', 'supervisor'):
        assert int((STAGE / (filename + '.exit')).read_text()) == 0
    post = json.loads((STAGE / 'source-postverify.json').read_text())
    assert all(post.values())
    policy = json.loads((STAGE / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM'
    assert all(policy[name] == '/opt/gfx-deps/dpcpp/bin/icpx' for name in ('CXX', 'SYCL', 'SYCL_host'))
    assert policy['C'] == '/opt/gfx-deps/dpcpp/bin/icx'
    manifest = json.loads((STAGE / 'source-native-package-manifest.json').read_text())
    site = Path(manifest['site_root'])
    assert manifest['normal_import_exit'] == 0 and not manifest['GPU_started']
    expected = {'libsgl-ops-sycl-' + name + '.so' for name in
                ('TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout', 'TripleOps')}
    assert set(manifest['ABI']) == expected
    hashes = {}
    for filename, record in manifest['ABI'].items():
        path = site / 'sgl_kernel' / filename
        assert digest(path) == record['fresh_sha256']
        assert record['old_sha256'] != record['fresh_sha256']
        assert record['required_original_common_ops_symbols']
        assert manifest['mapped_native_sha256'][str(path)] == record['fresh_sha256']
        hashes[filename] = record['fresh_sha256']
    for filename, expected_sha in manifest['mapped_native_sha256'].items():
        assert digest(Path(filename)) == expected_sha
    helper = json.loads((STAGE / 'fresh-helper-type-verification.json').read_text())
    assert helper['BF16_silu_actual_accumulator_type'] == 'float'
    assert helper['TripleOps_SHA256'] == hashes['libsgl-ops-sycl-TripleOps.so']
    assert not helper['GPU_initialized'] and helper['precision_not_run']
    for pid in (3291062, 3291063, 3291085):
        assert not Path(f'/proc/{pid}').exists()
    resource = json.loads((STAGE / 'native-build.resource-summary.json').read_text())
    assert resource['actual_exit'] == 0 and resource['drained']
    assert resource['peak_tree_rss_bytes'] < 16 * 1024 ** 3
    assert resource['peak_file_subtree_rss_bytes'] < 8 * 1024 ** 3
    report = {'actual_exit': 0, 'all_stage_phase_exits_zero': True,
              'fresh_four_native_installed_and_mapped_SHA_verified': hashes,
              'BF16_silu_actual_accumulator_type': 'float',
              'native_build_resource_summary': resource,
              'normal_original_registration_and_other_native_are_not_fresh_build_proof': True,
              'CPU_only_audit': True, 'GPU_precision_or_performance_acceptance': False,
              'source_manifest_sha256': digest(STAGE / 'source-native-package-manifest.json')}
    with (ROOT / 'ROUTING_CURRENT_SOURCE_MAIN_PACKAGE_AUDIT_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'fresh_tested_native': len(hashes),
                      'actual_BF16_silu_accumulator': 'float', 'GPU_precision_acceptance': False}))


if __name__ == '__main__':
    main()

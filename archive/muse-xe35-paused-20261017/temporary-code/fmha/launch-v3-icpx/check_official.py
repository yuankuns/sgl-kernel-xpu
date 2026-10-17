import argparse
import hashlib
import json
from pathlib import Path
import re


from settings import ART, CANONICAL_VERSION, FREEZE, STAGE
DEFAULT_FEATURES = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON',
                    'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def local_path(value):
    path = Path(value)
    if str(ART).startswith('/root/'):
        return Path(str(path).replace('/home/syk/', '/root/', 1))
    return Path(str(path).replace('/root/', '/home/syk/', 1))


def canonical_preflight():
    manifest_file = STAGE / 'source-binary-manifest.json'
    manifest = json.loads(manifest_file.read_text())
    complete = json.loads((STAGE / 'build-stage-complete.json').read_text())
    exact_freeze = json.loads(FREEZE.read_text())
    policy = json.loads((STAGE / 'compiler-policy-verified.json').read_text())
    assert policy['compiler_id'] == 'IntelLLVM' and policy['no_GNU_command'] is True
    for key, expected_compiler in {'cxx': 'icpx', 'sycl': 'icpx', 'host': 'icpx', 'c': 'icx'}.items():
        assert policy[key] == f'/opt/gfx-deps/dpcpp/bin/{expected_compiler}', key
    required_flags = {'-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen'}
    assert required_flags <= set(policy['required_flags'])
    cache_text = (STAGE / 'CMakeCache.txt').read_text()
    cache = {key: value for key, value in re.findall(r'^([^/#\n:]+):[^=\n]*=(.*)$', cache_text, re.MULTILINE)}
    for key, value in {'CMAKE_CXX_COMPILER': policy['cxx'], 'CMAKE_C_COMPILER': policy['c'],
                       'SYCL_COMPILER': policy['sycl'], 'SYCL_HOST_COMPILER': policy['host']}.items():
        assert cache[key] == value, key
    postverify = json.loads((STAGE / 'source-readonly-postverify.json').read_text())
    assert postverify['verified'] is True
    for name in ('configure.exit', 'supervisor.exit', 'default-all-build.exit'):
        assert (STAGE / name).read_text().strip() == '0', name
    site = (STAGE / 'site').resolve()
    assert (STAGE / 'default-all-build.exit').read_text().strip() == '0'
    assert manifest['build_evidence']['exit'] == complete['default_all_exit'] == manifest['install_actual_exit'] == 0
    assert complete['fresh_current_graph_install'] is True
    assert complete['compiler_policy'] == 'IntelLLVM C++ and SYCL icpx; C icx; no GCC host'
    assert local_path(manifest['site']).resolve() == site
    assert manifest['source']['head'] == exact_freeze['head'] == '833c95208dccd2af5628c121222f91186a5b2de7'
    assert local_path(manifest['source']['source']).name == 'muse-fmha-xe35'
    assert complete['source_digest'] == manifest['source']['source_digest']
    assert manifest['source']['source_digest'] == exact_freeze['source_digest']
    assert set(manifest['source']['files']) == set(exact_freeze['files'])
    for name, record in exact_freeze['files'].items():
        assert manifest['source']['files'][name]['sha256'] == record['sha256'], name
    assert manifest['cutlass_pin'] == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
    assert manifest['default_features'] == DEFAULT_FEATURES
    for name, record in (manifest['installed_files'] | manifest['python_snapshot_files']).items():
        path = (site / name).resolve()
        assert site in path.parents and digest(path) == record['sha256'], name
    expected = {name for name in manifest['current_native_targets']}
    actual = {path.name for path in (site / 'sgl_kernel').glob('*.so')}
    assert actual == expected == set(manifest['binaries']), (actual - expected, expected - actual)
    assert not any('flash_attentionXe35' in name for name in actual), 'diagnostic AFTER bridge forbidden'
    required = {'libsgl-ops-sycl-entry.so', 'libsgl-ops-sycl-flash_attention.so',
                'libsgl-ops-sycl-upstream_adapter.so', 'libsgl-ops-sycl-fmha_decode_64_bf16.so',
                'libsgl-ops-sycl-fmha_prefill_64_bf16.so'}
    assert required <= actual, required - actual
    assert any(name.startswith('common_ops.') for name in actual)
    return manifest, manifest_file, site / 'sgl_kernel', required


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    manifest, manifest_file, package_site, required = canonical_preflight()
    if args.preflight_only:
        print(json.dumps({'canonical_preflight': 'PASS', 'manifest_sha256': digest(manifest_file)}))
        return
    import torch
    import sgl_kernel

    assert Path(sgl_kernel.__file__).resolve().parent == package_site
    assert not torch.xpu.is_initialized(), 'normal import readiness must not initialize GPU before barrier'
    loaded = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        name = line.split()[-1]
        if 'common_ops.' not in name and 'libsgl-ops-' not in name and 'libsgl_jit' not in name:
            continue
        path = Path(name).resolve()
        assert path.parent == package_site, name
        assert not any(word in name for word in ('diagnostic', 'site-after', 'port_payload')), name
        expected = manifest['binaries'][path.name]
        assert path == (package_site.parent / expected['path']).resolve(), name
        assert digest(path) == expected['sha256'], name
        loaded[name] = expected['sha256']
    assert required <= {Path(name).name for name in loaded}
    assert any(Path(name).name.startswith('common_ops.') for name in loaded)
    report = {'acceptance': 'canonical_icpx_default_full_normal_import_only_gpu_validation_pending',
              'canonical_version': CANONICAL_VERSION,
              'compiler_policy_sha256': digest(STAGE / 'compiler-policy-verified.json'),
              'package': str(sgl_kernel.__file__), 'gpu_lease': 5, 'runtime_cpus': '16-31',
              'canonical_manifest_sha256': digest(manifest_file),
              'source_digest': manifest['source']['source_digest'],
              'frozen_source_digest': json.loads(FREEZE.read_text())['source_digest'],
              'build_provenance': 'main independent icpx pool real default-all and current-graph fresh install; no GNU host or historical object copies',
              'gpu_initialized': torch.xpu.is_initialized(), 'schema': str(torch.ops.sgl_kernel.fwd.default._schema),
              'loaded_libraries': loaded, 'diagnostic_AFTER_bridge': False}
    (STAGE / 'normal-import-readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

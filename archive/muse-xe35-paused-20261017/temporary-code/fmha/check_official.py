import argparse
import hashlib
import json
from pathlib import Path


ART = Path(__file__).resolve().parent
STAGE = ART / 'canonical-v2'
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
    frozen = json.loads((ART / 'frozen-source-sha256.json').read_text())
    exact_freeze = json.loads((ART / 'canonical-source-freeze-v2.json').read_text())
    site = (STAGE / 'site').resolve()
    assert (STAGE / 'default-all-build.exit').read_text().strip() == '0'
    assert manifest['build_evidence']['exit'] == complete['default_all_exit'] == manifest['install_actual_exit'] == 0
    assert complete['fresh_current_graph_install'] is True
    assert local_path(manifest['site']).resolve() == site
    assert manifest['source']['head'] == frozen['official_base']
    assert local_path(manifest['source']['source']).name == 'muse-fmha-xe35'
    assert complete['source_digest'] == manifest['source']['source_digest']
    assert manifest['source']['source_digest'] == exact_freeze['source_digest']
    assert set(manifest['source']['files']) == set(exact_freeze['files'])
    for name, record in exact_freeze['files'].items():
        assert manifest['source']['files'][name]['sha256'] == record['sha256'], name
    assert manifest['cutlass_pin'] == frozen['public_cutlass_pin']
    assert manifest['default_features'] == DEFAULT_FEATURES
    for name, expected in frozen['files'].items():
        assert manifest['source']['files'][name]['sha256'] == expected, name
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
    report = {'acceptance': 'canonical_default_full_normal_import_only_gpu_validation_pending',
              'package': str(sgl_kernel.__file__), 'gpu_lease': 5, 'runtime_cpus': '16-31',
              'canonical_manifest_sha256': digest(manifest_file),
              'source_digest': manifest['source']['source_digest'],
              'frozen_source_digest': json.loads((ART / 'frozen-source-sha256.json').read_text())['aggregate_source_sha256'],
              'build_provenance': 'main fixed-path canonical real default-all incremental plus current-graph fresh install; not independent clean',
              'gpu_initialized': torch.xpu.is_initialized(), 'schema': str(torch.ops.sgl_kernel.fwd.default._schema),
              'loaded_libraries': loaded, 'diagnostic_AFTER_bridge': False}
    (ART / 'official-ready.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

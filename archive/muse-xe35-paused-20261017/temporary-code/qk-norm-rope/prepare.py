import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


root = Path(__file__).resolve().parent
trees = root.parents[1]
source = trees.parent / 'sgl-kernel-xpu/muse-spark-qk-norm-rope-main-20261014'
evidence = trees.parent / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
target = trees / 'pr/muse-qk-norm-rope-xe35'
expected_source = 'feed315712ea91e6becbb983f3a4d56ac0cc3b96'
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == expected_source
assert subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip() == '833c95208dccd2af5628c121222f91186a5b2de7'
provenance = json.loads((evidence / 'provenance.json').read_text())
manifest_path = evidence / 'after/site-manifest.json'
assert digest_xe35(manifest_path) == provenance['after']['site_manifest_sha256']
manifest = json.loads(manifest_path.read_text())
for record in manifest:
    assert digest_xe35(evidence / record['path']) == record['sha256'], record['path']
source_files = {}
for relative, expected in provenance['after']['source_hashes'].items():
    path = source / relative
    assert digest_xe35(path) == expected, relative
    assert path.read_bytes() == subprocess.check_output(['git', '-C', str(source), 'show', f'{expected_source}:{relative}'])
    source_files[relative] = expected
for tokens in (8, 32768):
    assert (evidence / f'after/accuracy-t{tokens}.exit').read_text().strip() == '0'
assert (root / 'build-full.exit').read_text().strip() == '0', 'Default-feature full build must pass before staging'
cache = (target / 'build/CMakeCache.txt').read_text()
default_features = {'USE_MOE': 'ON', 'USE_FMHA': 'ON', 'USE_GDN': 'ON', 'USE_MLA': 'ON', 'USE_QSA': 'ON', 'USE_MLA_SPARSE_FUSED': 'OFF', 'USE_SYCL_JIT': 'OFF'}
for feature, value in default_features.items():
    assert f'{feature}:BOOL={value}\n' in cache, f'Not a default-feature full build: {feature}'
assert 'DPCPP_SYCL_TARGET:STRING=cri\n' in cache
assert 'FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL:PATH=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56\n' in cache
wheels = list((root / 'wheels').glob('*.whl'))
assert len(wheels) == 1, wheels
wheel = wheels[0]
site = root / 'port/site'
assert not site.exists(), 'Do not overwrite an existing isolated package'
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(site)
package = site / 'sgl_kernel'
assert len(list(package.glob('common_ops*.so'))) == 1
assert (package / 'libsgl-ops-sycl-FusedQKNormRope_xe35.so').is_file()
for path in sorted((target / 'python/sgl_kernel').rglob('*.py')):
    relative = path.relative_to(target / 'python/sgl_kernel')
    assert digest_xe35(path) == digest_xe35(package / relative), f'Official wheel Python file differs from target: {relative}'
port_manifest = [{'path': str(path.relative_to(root)), 'sha256': digest_xe35(path)} for path in sorted(site.rglob('*')) if path.is_file()]
(root / 'port/site-manifest.json').write_text(json.dumps(port_manifest, indent=2) + '\n')
port_files = subprocess.check_output(['git', '-C', str(target), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
port_hashes = {relative: digest_xe35(target / relative) for relative in port_files if (target / relative).is_file()}
record = {
    'source_commit': expected_source,
    'target_base': '833c95208dccd2af5628c121222f91186a5b2de7',
    'source_site': str(evidence / 'after/site'),
    'source_manifest': str(manifest_path),
    'source_files_verified': len(manifest),
    'source_qk_library_sha256': provenance['after']['qk_library_sha256'],
    'source_hashes': source_files,
    'historical_accuracy': {str(tokens): {'log': str(evidence / f'after/accuracy-t{tokens}.log'), 'sha256': digest_xe35(evidence / f'after/accuracy-t{tokens}.log')} for tokens in (8, 32768)},
    'source_site_known_difference': json.loads((evidence / 'source-match.json').read_text()),
    'port_site': str(site),
    'port_hashes': port_hashes,
    'wheel': str(wheel),
    'wheel_sha256': digest_xe35(wheel),
    'port_manifest_sha256': digest_xe35(root / 'port/site-manifest.json'),
    'default_features': default_features,
    'cmake_cache_sha256': digest_xe35(target / 'build/CMakeCache.txt'),
}
(root / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'source_verified_files': len(manifest), 'fresh_port_files': len(port_manifest), 'wheel_sha256': record['wheel_sha256']}, indent=2))

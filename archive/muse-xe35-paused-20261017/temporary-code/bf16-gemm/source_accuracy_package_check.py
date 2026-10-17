import hashlib
import json
import subprocess
import time
from pathlib import Path


VERSION = 'source-native-icpx-v1'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def preflight(root, grant_file, gpu):
    grant_path = Path(grant_file)
    grant = json.loads(grant_path.read_text())
    stage = root / 'fresh-source-icpx-v1'
    manifest_file = stage / 'source-native-binary-manifest.json'
    manifest = json.loads(manifest_file.read_text())
    pre = json.loads((stage / 'source-preverify.json').read_text())
    post = json.loads((stage / 'source-postverify.json').read_text())
    policy = json.loads((stage / 'compiler-policy-verified.json').read_text())
    for name in ['configure.exit', 'native-build.exit', 'container.exit', 'supervisor.exit']:
        if (stage / name).read_text().strip() != '0':
            raise RuntimeError('Fresh SOURCE actual exit0 required: ' + name)
    if grant.get('version') != VERSION or grant.get('scope') != 'SOURCE accuracy only' or grant.get('module') != root.name:
        raise RuntimeError('Only a fresh SOURCE-only accuracy grant is accepted')
    if grant.get('gpu') != gpu or grant.get('coordinator_release') is not True:
        raise RuntimeError('No accuracy grant for this card/module')
    if grant.get('fresh_baseline_native_sha256') != manifest['native_SHA256'] or grant.get('source_manifest_sha256') != digest(manifest_file):
        raise RuntimeError('Grant must bind the exact fresh SOURCE native/manifest')
    if not post.get('verified') or not post['epoch'] < grant.get('epoch', 0) <= time.time() + 60:
        raise RuntimeError('Fresh epoch after successful SOURCE postverify is required')
    if post['source_preverify_sha256'] != digest(stage / 'source-preverify.json') or post['source_HEAD'] != manifest['source_HEAD']:
        raise RuntimeError('SOURCE readonly pre/post proof mismatch')
    if manifest['source_preverify_sha256'] != digest(stage / 'source-preverify.json') or manifest['compiler_policy_sha256'] != digest(stage / 'compiler-policy-verified.json'):
        raise RuntimeError('SOURCE build/flags proof hash mismatch')
    expected = '/opt/gfx-deps/dpcpp/bin/icpx'
    if policy.get('compiler_id') != 'IntelLLVM' or any(policy.get(key) != expected for key in ['CXX', 'SYCL', 'SYCL_host', 'device_link', 'native_link']):
        raise RuntimeError('Fresh tested SOURCE native must use genuine icpx')
    if not policy.get('kernel_compile_flags_byte_equal_historical') or not policy.get('original_precision_AOT_GRF_preserved'):
        raise RuntimeError('Original production flags must remain unchanged')
    if digest(stage / 'build/CMakeCache.txt') != policy['CMakeCache_sha256'] or digest(stage / 'build/build.ninja') != policy['build_ninja_sha256']:
        raise RuntimeError('Actual compiler cache/Ninja changed')
    if manifest['source_pin'] != 'e87870bb7aacf83bfb3be9da685f148316b8df23':
        raise RuntimeError('Genuine original SOURCE dependency pin required')
    for name, checksum in pre['snapshot_files'].items():
        if digest(stage / 'source' / name) != checksum:
            raise RuntimeError('Frozen SOURCE/header bytes changed: ' + name)
    historical = root / 'source/site/sgl_kernel'
    fresh = stage / 'site/sgl_kernel'
    for package, expected_files in [(historical, manifest['SOURCE_original_files']), (fresh, manifest['site_files'])]:
        actual_names = {str(path.relative_to(package)) for path in package.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        if actual_names != set(expected_files):
            raise RuntimeError('Full genuine SOURCE package inventory mismatch')
        for name, checksum in expected_files.items():
            if digest(package / name) != checksum:
                raise RuntimeError('Full genuine SOURCE package hash mismatch: ' + name)
    native_name = manifest['native_SONAME']
    if manifest['changed_package_files'] != [native_name]:
        raise RuntimeError('Only freshly compiled genuine SOURCE native may be replaced')
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(fresh / native_name)], text=True).split()
    if not all(name in symbols for name in manifest['native_ABI_export_checked']):
        raise RuntimeError('Fresh SOURCE actual ABI export failure; no glue allowed')
    dynamic = subprocess.check_output(['readelf', '-d', str(fresh / native_name)], text=True)
    if 'Library soname: [' + native_name + ']' not in dynamic:
        raise RuntimeError('Fresh SOURCE SONAME changed')
    return manifest, {'file': str(grant_path), 'sha256': digest(grant_path), 'grant': grant,
                      'manifest_sha256': digest(manifest_file), 'postverify_sha256': digest(stage / 'source-postverify.json')}


def mapped_provenance(package, expected_files, native):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'common_ops.' in line or 'libsgl-ops-' in line or 'libsgl_jit' in line:
            path = Path(line.split()[-1]).resolve()
            if package not in path.parents:
                raise RuntimeError('Mapped genuine SOURCE DSO is outside this normal package: ' + line)
            paths.add(path)
    if package / native not in paths or not any(path.name.startswith('common_ops.') for path in paths):
        raise RuntimeError('Real normal package/native initialization required')
    hashes = {str(path): digest(path) for path in sorted(paths)}
    for path, checksum in hashes.items():
        if expected_files.get(str(Path(path).relative_to(package))) != checksum:
            raise RuntimeError('Actual mapped SOURCE DSO SHA mismatch: ' + path)
    return {'loaded_libraries': hashes, 'all_mapped_sgl_dso_hashes_verified': True,
            'tested_native_sha256': hashes[str(package / native)]}

import argparse
import time
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


VERSION = 'canonical-v3-icpx'
CXX = '/opt/gfx-deps/dpcpp/bin/icpx'
CC = '/opt/gfx-deps/dpcpp/bin/icx'
REQUIRED_FLAGS = {'-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz', '-fsycl-targets=spir64_gen'}


def runtime_path(value, root):
    if str(root).startswith('/root/'):
        value = value.replace('/home/syk/', '/root/', 1)
    else:
        value = value.replace('/root/', '/home/syk/', 1)
    return Path(value).resolve()


def file_set_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_compiler_policy(policy, flags):
    if policy.get('compiler_id') != 'IntelLLVM' or policy.get('no_GNU_command') is not True:
        raise RuntimeError('Real IntelLLVM / no GNU command proof is required')
    expected = {'cxx': CXX, 'sycl': CXX, 'host': CXX, 'c': CC}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise RuntimeError('CXX/SYCL/host must all be icpx; C must be icx')
    if not REQUIRED_FLAGS.issubset(set(policy.get('required_flags', []))):
        raise RuntimeError('Production precision compiler flags are missing')
    keys = {'CMAKE_CXX_COMPILER': CXX, 'CMAKE_C_COMPILER': CC,
            'SYCL_COMPILER': CXX, 'SYCL_EXECUTABLE': CXX, 'SYCL_HOST_COMPILER': CXX}
    for key, value in keys.items():
        if flags.get(key, {}).get('value') != value:
            raise RuntimeError('Manifest compiler configuration mismatch: ' + key)


def canonical_preflight(root, version):
    if version != VERSION:
        raise RuntimeError('Only canonical-v3-icpx is accepted; v1/v2 packages are evidence only')
    stage = root / VERSION
    manifest_file = stage / 'source-binary-manifest.json'
    manifest = json.loads(manifest_file.read_text())
    complete = json.loads((stage / 'build-stage-complete.json').read_text())
    post = json.loads((stage / 'source-readonly-postverify.json').read_text())
    snapshot = json.loads((stage / 'source-snapshot.json').read_text())
    frozen_file = root / 'canonical-source-freeze-v3-icpx.json'
    frozen = json.loads(frozen_file.read_text())
    policy_file = stage / 'compiler-policy-verified.json'
    policy = json.loads(policy_file.read_text())
    for name in ['configure.exit', 'default-all-build.exit', 'supervisor.exit']:
        if (stage / name).read_text().strip() != '0':
            raise RuntimeError('Real configure/default all/supervisor exit0 required: ' + name)
    if manifest.get('install_actual_exit') != 0 or complete.get('default_all_exit') != 0 or complete.get('fresh_current_graph_install') is not True:
        raise RuntimeError('Current all and fresh install must succeed')
    if 'CANONICAL_INSTALL_ACTUAL_EXIT=0' not in (stage / 'install.log').read_text():
        raise RuntimeError('Actual install exit0 record is missing')
    if post.get('verified') is not True:
        raise RuntimeError('Readonly source postverify must succeed')
    if runtime_path(post['snapshot_manifest'], root) != (stage / 'source-snapshot.json').resolve():
        raise RuntimeError('Postverify belongs to another snapshot')
    if frozen.get('version') != VERSION:
        raise RuntimeError('A new v3-icpx reviewed source freeze is required')
    prior = json.loads((root / 'canonical-source-freeze-v2.json').read_text())
    changed = sorted(name for name in frozen['files'].keys() & prior['files'].keys() if frozen['files'][name] != prior['files'][name])
    if frozen.get('prior_version') != 'canonical-v2' or set(frozen['files']) != set(prior['files']) or changed != ['cmake/BuildFlags.cmake']:
        raise RuntimeError('v2 to v3 must contain only the reviewed BuildFlags compiler prerequisite')
    files = manifest['source']['files']
    source_digest = file_set_digest(files)
    if not (source_digest == manifest['source']['source_digest'] == complete['source_digest'] == snapshot['source_digest'] == frozen['source_digest']):
        raise RuntimeError('Freeze/snapshot/default all/install source digests do not match')
    if not (files == snapshot['files'] == frozen['files']):
        raise RuntimeError('Reviewed freeze bytes differ from installed source manifest')
    if not snapshot.get('verified') or manifest['source']['head'] != '833c95208dccd2af5628c121222f91186a5b2de7':
        raise RuntimeError('Verified official upstream snapshot is required')
    if runtime_path(manifest['snapshot'], root).parent.name != 'canonical-icpx-pool':
        raise RuntimeError('Old GNU/icx canonical pool is forbidden')
    module_tree = 'muse-' + root.name + '-xe35'
    if Path(manifest['source']['source']).name != module_tree or Path(post['source']).name != module_tree:
        raise RuntimeError('Snapshot/postverify belongs to another module')
    validate_compiler_policy(policy, manifest['flags'])
    if digest(stage / 'CMakeCache.txt') != manifest['cache_sha256'] or digest(stage / 'build.ninja') != manifest['ninja_sha256']:
        raise RuntimeError('Archived actual compiler configuration differs from install evidence')
    if manifest['cutlass_pin'] != '7b56b936b87f799f80dde60c2fd8d11c1128cebb':
        raise RuntimeError('Public exact pin is required')
    expected_site = (stage / 'site').resolve()
    if runtime_path(manifest['site'], root) != expected_site:
        raise RuntimeError('Manifest points outside the new v3 site')
    expected_files = manifest['installed_files'] | manifest['python_snapshot_files']
    for name, record in expected_files.items():
        path = expected_site / name
        if expected_site not in path.resolve().parents or digest(path) != record['sha256']:
            raise RuntimeError('Current v3 package hash mismatch: ' + name)
    manifest['checker_proofs'] = {'compiler_policy_sha256': digest(policy_file),
                                  'source_freeze_sha256': digest(frozen_file),
                                  'postverify_sha256': digest(stage / 'source-readonly-postverify.json'),
                                  'source_digest': source_digest, 'source_version': VERSION,
                                  'real_compiler_policy_verified': True, 'post_epoch': post['epoch']}
    return manifest, manifest_file, expected_site / 'sgl_kernel'


def validate_grant(grant, root, manifest, gpu, phase):
    if grant.get('stage') != VERSION or grant.get('source_digest') != manifest['source']['source_digest']:
        raise RuntimeError('Fresh grant must bind canonical-v3-icpx and its exact source digest')
    if grant.get('module') != root.name or grant.get('gpu') != gpu or grant.get('phase') != phase or grant.get('coordinator_release') is not True:
        raise RuntimeError('No coordinator grant for this module/GPU/phase')
    epoch = grant.get('epoch')
    if not isinstance(epoch, (int, float)) or isinstance(epoch, bool) or not manifest['checker_proofs']['post_epoch'] < epoch <= time.time() + 60:
        raise RuntimeError('Grant must be a fresh epoch after successful postverify')
    if grant.get('source_binary_manifest_sha256') != digest(root / VERSION / 'source-binary-manifest.json'):
        raise RuntimeError('Grant must bind the current source-binary manifest SHA')
    if root.name == 'moe-permutation' and phase == 'performance' and grant.get('previous_bf16_drained') is not True:
        raise RuntimeError('MoE GPU4 performance must wait for explicit BF16-drained confirmation')


def grant_preflight(root, grant_file, accuracy_only):
    if not grant_file:
        raise RuntimeError('Missing fresh grant; stop before GPU lock/import')
    path = Path(grant_file)
    grant = json.loads(path.read_text())
    manifest, manifest_file, site = canonical_preflight(root, VERSION)
    gpu = 10 if accuracy_only else 4
    validate_grant(grant, root, manifest, gpu, 'accuracy' if accuracy_only else 'performance')
    return manifest, manifest_file, site, {'path': str(path.resolve()), 'sha256': digest(path), 'grant': grant}


def main():
    parser = argparse.ArgumentParser(description='CPU-only v3 icpx/source/package/grant gate; no Torch or GPU lock')
    parser.add_argument('--grant-file', required=True)
    parser.add_argument('--accuracy-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    manifest, manifest_file, site, grant = grant_preflight(root, args.grant_file, args.accuracy_only)
    print(json.dumps({'source_version': VERSION, 'source_digest': manifest['source']['source_digest'],
                      'manifest_sha256': digest(manifest_file), 'site': str(site),
                      'compiler_proofs': manifest['checker_proofs'], 'grant': grant, 'GPU_started': False}))


def runtime_provenance(root, label, package_site, manifest=None, manifest_file=None):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if 'common_ops.' in line or 'libsgl-ops-' in line or 'libsgl_jit' in line:
            path = Path(fields[-1])
            if package_site not in path.resolve().parents:
                raise RuntimeError('Mapped native binary is outside the real package: ' + line)
            paths.add(path)
    if not any(path.name.startswith('common_ops.') for path in paths):
        raise RuntimeError('Normal package initialization did not map common_ops')
    hashes = {str(path): digest(path) for path in sorted(paths)}
    if label == 'source':
        baseline = json.loads((root / 'source-provenance.json').read_text())
        for path, checksum in hashes.items():
            if baseline['binary_sha256'].get(Path(path).name) != checksum:
                raise RuntimeError('Mapped SOURCE binary differs from the validated baseline: ' + path)
    if label == 'port':
        expected = {str((package_site.parent / record['path']).resolve()): record['sha256'] for record in manifest['binaries'].values()}
        for path, checksum in hashes.items():
            if expected.get(str(Path(path).resolve())) != checksum:
                raise RuntimeError('Mapped canonical native DSO differs from current install graph: ' + path)
        if any('port_payload' in path or 'diagnostic' in path for path in hashes):
            raise RuntimeError('Diagnostic AFTER payload is forbidden for official integration')
    return {'loaded_libraries': hashes, 'all_mapped_sgl_dso_hashes_verified': True,
            'canonical_manifest_sha256': digest(manifest_file) if manifest_file else None,
            'source_provenance_sha256': digest(root / 'source-provenance.json') if label == 'source' else None}


if __name__ == '__main__':
    main()

import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def canonical_preflight(root, version):
    stage = root / version
    manifest_file = stage / 'source-binary-manifest.json'
    manifest = json.loads(manifest_file.read_text())
    complete = json.loads((stage / 'build-stage-complete.json').read_text())
    if manifest.get('install_actual_exit') != 0 or complete.get('default_all_exit') != 0:
        raise RuntimeError('Canonical default all and fresh install must both be successful')
    if (stage / 'default-all-build.exit').read_text().strip() != '0':
        raise RuntimeError('Canonical default all has no real exit0')
    expected_site = (stage / 'site').resolve()
    recorded_site = Path(manifest['site'].replace('/home/syk/', '/root/', 1)).resolve()
    if recorded_site != expected_site:
        raise RuntimeError('Canonical package manifest points to another site')
    if complete['source_digest'] != manifest['source']['source_digest']:
        raise RuntimeError('Canonical build/source digest mismatch')
    if manifest['source']['head'] != '833c95208dccd2af5628c121222f91186a5b2de7':
        raise RuntimeError('Canonical package is not based on the official upstream revision')
    module_tree = 'muse-' + root.name + '-xe35'
    if Path(manifest['source']['source']).name != module_tree:
        raise RuntimeError('Canonical package belongs to another module')
    expected_files = manifest['installed_files'] | manifest['python_snapshot_files']
    for name, record in expected_files.items():
        path = expected_site / name
        if expected_site not in path.resolve().parents or digest(path) != record['sha256']:
            raise RuntimeError('Current canonical package hash mismatch: ' + name)
    return manifest, manifest_file, expected_site / 'sgl_kernel'


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

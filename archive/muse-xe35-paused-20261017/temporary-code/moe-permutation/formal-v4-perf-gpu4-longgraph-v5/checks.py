from release import execute_frozen, maintenance_gate

execute_frozen('checks.py', globals())
EVIDENCE_DIRECTORY = 'formal-v4-perf-gpu4-longgraph-v5'
ROUND = ROOT / EVIDENCE_DIRECTORY
RUN_PREFIX = VERSION + '-' + EVIDENCE_DIRECTORY.replace('formal-v4-perf-', 'formal-')
_original_driver_hashes = driver_hashes
_original_write_json = write_json
_original_grant_preflight = grant_preflight


def driver_hashes():
    files, identity = _original_driver_hashes()
    inventory = json.loads((ROUND / 'frozen-implementation-manifest.json').read_text())
    for name, expected in inventory['files_sha256'].items():
        if digest(ROOT / inventory['version'] / name) != expected:
            raise RuntimeError('Frozen runtime driver changed: ' + name)
        files['frozen/' + name] = expected
    files['frozen-implementation-manifest.json'] = digest(ROUND / 'frozen-implementation-manifest.json')
    files['main_external_maintenance_snapshot'] = digest(ROOT.parent / 'EXTERNAL_MAINTENANCE_SNAPSHOT_20261017_v1.json')
    files['release-authorization.json'] = digest(ROUND / 'release-authorization.json')
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return files, identity


def write_json(path, value):
    if path == ROUND / 'fresh-grant.json':
        record = maintenance_gate()
        value.update({'maintenance_release': True,
                      'maintenance_snapshot_sha256': record['maintenance_snapshot_sha256'],
                      'loaded_stack': {'running_kernel': record['running_kernel'], 'loaded_modules': record['loaded_modules']},
                      'release_authorization_sha256': digest(ROUND / 'release-authorization.json'),
                      'superseded_held_grant_not_consumed': True})
    _original_write_json(path, value)


def grant_preflight(root, grant_file, accuracy_only):
    result = _original_grant_preflight(root, grant_file, accuracy_only)
    record = maintenance_gate()
    grant = json.loads(Path(grant_file).read_text())
    expected = {'maintenance_release': True, 'maintenance_snapshot_sha256': record['maintenance_snapshot_sha256'],
                'loaded_stack': {'running_kernel': record['running_kernel'], 'loaded_modules': record['loaded_modules']},
                'release_authorization_sha256': digest(ROUND / 'release-authorization.json'),
                'superseded_held_grant_not_consumed': True}
    for name, value in expected.items():
        if grant.get(name) != value:
            raise RuntimeError('Maintenance release binding mismatch: ' + name)
    return result

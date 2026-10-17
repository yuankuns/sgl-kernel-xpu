from release import execute_frozen, maintenance_gate

execute_frozen('checks.py', globals())
SCOPE = 'BF independent clean confirmation original reviewed single BAAB/all12paths'
_original_driver_hashes = driver_hashes
_original_write_json = write_json
_original_grant_preflight = grant_preflight


def driver_hashes():
    files, identity = _original_driver_hashes()
    inventory = json.loads((ROUND / 'frozen-implementation-manifest.json').read_text())
    for name, expected in inventory['files_sha256'].items():
        if digest(ROOT / inventory['version'] / name) != expected:
            raise RuntimeError('Frozen original BF driver changed: ' + name)
        files['frozen/' + name] = expected
    for name in ['frozen-implementation-manifest.json', 'release-authorization.json']:
        files[name] = digest(ROUND / name)
    for name, expected in json.loads((ROUND / 'release-authorization.json').read_text())['FM_release_proofs_sha256'].items():
        files['new_release/' + name] = expected
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return files, identity


def release_bindings(record):
    for name in ['CPU-protocol.exit', 'CPU-source-provider.exit', 'cpu-gate.exit']:
        if (ROUND / name).read_text().strip() != '0':
            raise RuntimeError('Fresh BF CPU gate not actual0: ' + name)
    return {'FM_release_proofs_sha256': record['FM_release_proofs'],
            'maintenance_snapshot_sha256': record['maintenance_snapshot_sha256'],
            'loaded_stack': {'running_kernel': record['running_kernel'], 'loaded_modules': record['loaded_modules']},
            'release_authorization_sha256': digest(ROUND / 'release-authorization.json'),
            'CPU_protocol_gate_sha256': digest(ROUND / 'CPU_PROTOCOL_INPUT_INTERFACE.json'),
            'CPU_SOURCE_provider_gate_sha256': digest(ROUND / 'CPU_SOURCE_PROVIDER.json'),
            'SOURCE_precision_summary_sha256': digest(ROOT / 'source-native-icpx-v1-accuracy/summary.json'),
            'previous_MoE_drained_sha256': digest(ROOT.parent / 'moe-permutation/formal-v4-perf-gpu4-longgraph-v6/GPU-drained.json'),
            'old_BF_compiler_chronology_sha256': digest(ROOT / 'BF_NV_V5_TIMELINE_SIDECAR.json'),
            'calls': 20, 'repeats': 20, 'warmup': 5, 'minimum_BAAB_rounds': 1,
            'old_grants_not_consumed': True, 'MoE_performance_accepted': False}


def write_json(path, value):
    if path == ROUND / 'fresh-grant.json':
        record = maintenance_gate()
        if time.time() <= record['epoch']:
            raise RuntimeError('Maintenance epoch invalid')
        value.update(release_bindings(record))
        _original_write_json(ROUND / 'compiler-maintenance-pregrant.json', record)
    _original_write_json(path, value)


def grant_preflight(root, grant_file, accuracy_only):
    result = _original_grant_preflight(root, grant_file, accuracy_only)
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError('Original genuine BF preflight four-tuple not returned')
    record = maintenance_gate()
    grant = json.loads(Path(grant_file).read_text())
    for name, expected in release_bindings(record).items():
        if grant.get(name) != expected:
            raise RuntimeError('Fresh BF metadata mismatch: ' + name)
    _original_write_json(ROUND / ('compiler-maintenance-prelaunch.json' if '/home/' in str(ROUND) else 'compiler-maintenance-GPU-runtime.json'), record)
    return result

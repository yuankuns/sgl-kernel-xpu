import hashlib
import json
from pathlib import Path


ART = Path(__file__).resolve().parent
HOME = Path('/root') if str(ART).startswith('/root/') else Path('/home/syk')
ARCHIVE = HOME / 'worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017'
BASE = 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
REQUIRED = {'libsgl-ops-sycl-flash_attentionXe35.so',
            'libsgl-ops-sycl-flash_attentionXe35_decode_64_bf16.so',
            'libsgl-ops-sycl-flash_attentionXe35_prefill_64_bf16.so'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def preflight():
    source = json.loads((ARCHIVE / 'source-manifest.json').read_text())
    before = next(record for record in source['sources'] if record['side'] == 'before')
    assert before['head'] == BASE and before['tracked_source_clean'] is True
    assert source['workload'] == {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
                                  'dtype': 'BF16', 'heads_q': 128, 'heads_kv': 16, 'head_dim': 64, 'page_size': 64}
    package = ARCHIVE / 'site-before/sgl_kernel'
    records = json.loads((ARCHIVE / 'binary-manifest-before.json').read_text())
    assert len(records) == 216
    hashes = {}
    for record in records:
        name = Path(record['path']).name
        assert digest(package / name) == record['sha256'], name
        hashes[name] = record['sha256']
    for name in REQUIRED:
        assert digest(ARCHIVE / 'build-before/src' / name) == hashes[name], name
    evidence = json.loads((ARCHIVE / 'run-evidence.json').read_text())
    before_runs = [record for record in evidence['runs'] if record['source_head'] == BASE]
    assert before_runs
    for record in before_runs:
        assert record['exit'] == 0
        assert all(record['loaded_target_libraries_sha256'][name] == hashes[name] for name in REQUIRED)
    return package, hashes


def verify_loaded(package, hashes, required):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        name = line.split()[-1]
        if 'common_ops.' not in name and 'libsgl-ops-' not in name and 'libsgl_jit' not in name:
            continue
        path = Path(name).resolve()
        assert path.parent == package.resolve(), name
        assert digest(path) == hashes[path.name], name
        paths.add(path.name)
    assert required <= paths, required - paths
    assert any(name.startswith('common_ops.') for name in paths)


if __name__ == '__main__':
    package, hashes = preflight()
    print(json.dumps({'preopt_cpu_provenance': 'PASS', 'base_sha': BASE,
                      'package': str(package), 'manifest_files': len(hashes),
                      'fresh_production_libraries': {name: hashes[name] for name in sorted(REQUIRED)},
                      'scope': 'historical staged normal package; not canonical default-all fresh install'}))

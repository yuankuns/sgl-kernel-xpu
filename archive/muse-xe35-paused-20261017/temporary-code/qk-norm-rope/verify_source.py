import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
worktrees = root.parents[2]
source = worktrees / 'sgl-kernel-xpu/muse-spark-qk-norm-rope-main-20261014'
evidence = worktrees / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
dependency = root.parent / 'deps/sycl-tla-7b56'


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


source_git = ['git', '-c', f'safe.directory={source.resolve()}', '-C', str(source.resolve())]
dependency_git = ['git', '-c', f'safe.directory={dependency.resolve()}', '-C', str(dependency.resolve())]
commit = subprocess.check_output([*source_git, 'rev-parse', 'HEAD'], text=True).strip()
assert commit == 'feed315712ea91e6becbb983f3a4d56ac0cc3b96'
provenance = json.loads((evidence / 'provenance.json').read_text())
manifest = evidence / 'after/site-manifest.json'
assert digest_xe35(manifest) == provenance['after']['site_manifest_sha256']
records = json.loads(manifest.read_text())
for record in records:
    assert digest_xe35(evidence / record['path']) == record['sha256'], record['path']
for relative, expected in provenance['after']['source_hashes'].items():
    path = source / relative
    assert digest_xe35(path) == expected, relative
    assert path.read_bytes() == subprocess.check_output([*source_git, 'show', f'{commit}:{relative}'])
assert digest_xe35(evidence / 'after/site/sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so') == provenance['after']['qk_library_sha256']
assert subprocess.check_output([*dependency_git, 'rev-parse', 'HEAD'], text=True).strip() == '7b56b936b87f799f80dde60c2fd8d11c1128cebb'
assert not subprocess.check_output([*dependency_git, 'status', '--porcelain'], text=True).strip()
accuracy = {}
for tokens in (8, 32768):
    assert (evidence / f'after/accuracy-t{tokens}.exit').read_text().strip() == '0'
    log = evidence / f'after/accuracy-t{tokens}.log'
    record = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{') and 'violations' in line)
    assert record['violations'] == 0 and record['tail_equal'] and record['atol'] == record['rtol'] == 5e-3
    accuracy[str(tokens)] = {'log': str(log), 'log_sha256': digest_xe35(log), 'result': record}
result = {
    'source_commit': commit,
    'source_site_files_verified': len(records),
    'source_inputs_git_bytes_verified': len(provenance['after']['source_hashes']),
    'source_qk_binary_sha256': provenance['after']['qk_library_sha256'],
    'public_dependency_head': '7b56b936b87f799f80dde60c2fd8d11c1128cebb',
    'public_dependency_clean': True,
    'historical_after_accuracy_verified': accuracy,
    'known_full_package_source_difference': json.loads((evidence / 'source-match.json').read_text()),
}
(root / 'source-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ('historical_after_accuracy_verified', 'known_full_package_source_difference')}, indent=2))

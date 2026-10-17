import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-bf16-gemm-main-20261014')
OLD = Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/bf16-rebase-before-after-20261017')


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main_xe35():
    evidence = json.loads((OLD / 'provenance.json').read_text())
    variant = evidence['variants']['after']
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']).decode().strip() == variant['commit']
    for name, checksum in variant['source_sha256'].items():
        assert digest_xe35(SOURCE / name) == checksum, name
    stage = OLD / 'after/site/sgl_kernel'
    hashes = json.loads((OLD / 'after/sha256.json').read_text())
    for name, checksum in hashes.items():
        assert digest_xe35(OLD / 'after' / name) == checksum, name
    destination = ROOT / 'source/site/sgl_kernel'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / 'python/sgl_kernel', destination, dirs_exist_ok=True)
    for binary in stage.glob('*.so'):
        subprocess.run(['cp', '--reflink=auto', str(binary), str(destination / binary.name)], check=True)
    record = {'source_commit': variant['commit'], 'evidence_sha256': digest_xe35(OLD / 'provenance.json'),
              'historical_evidence': str(OLD), 'reuse_scope': evidence['binary_reuse_scope'],
              'source_variant': variant, 'new_runtime_validation_required': True,
              'site': str(destination.parent), 'binary_sha256': {path.name: digest_xe35(path) for path in destination.glob('*.so')}}
    (ROOT / 'source-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main_xe35()

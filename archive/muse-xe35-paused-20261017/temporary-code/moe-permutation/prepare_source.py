import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-permutation-only-main-20261015')
OLD = Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/permutation-scope-correction')


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main_xe35():
    assert (OLD / 'full-build.exit').read_text().strip() == '0'
    freeze = json.loads((OLD / 'source-freeze.json').read_text())
    mismatches = []
    for entry in freeze['files']:
        path = SOURCE / entry['path']
        if not path.is_file() or digest_xe35(path) != entry['sha256']:
            mismatches.append(entry['path'])
    assert not mismatches, mismatches
    stage = OLD / 'install/sgl_kernel'
    checked = {}
    for line in (OLD / 'full-built-sha256.txt').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        path = stage / Path(name).name
        assert digest_xe35(path) == checksum, name
        checked[path.name] = checksum
    destination = ROOT / 'source/site/sgl_kernel'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / 'python/sgl_kernel', destination, dirs_exist_ok=True)
    for binary in stage.glob('*.so'):
        subprocess.run(['cp', '--reflink=auto', str(binary), str(destination / binary.name)], check=True)
    record = {'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']).decode().strip(),
              'historical_evidence': str(OLD), 'source_freeze_sha256': digest_xe35(OLD / 'source-freeze.json'),
              'full_build_exit': 0, 'source_content_mismatches': mismatches, 'binary_sha256': checked,
              'new_runtime_validation_required': True, 'site': str(destination.parent)}
    (ROOT / 'source-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main_xe35()

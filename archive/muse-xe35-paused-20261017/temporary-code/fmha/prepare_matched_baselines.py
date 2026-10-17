import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ART = Path(__file__).resolve().parent
ARCHIVE = Path('/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017')
ROOT = ART / 'matched-native-icpx-v1'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    manifest = json.loads((ARCHIVE / 'source-manifest.json').read_text())
    ROOT.mkdir(exist_ok=False)
    dependency = Path(manifest['cutlass']['source'])
    for name, expected in manifest['cutlass']['files_sha256'].items():
        assert digest(dependency / name) == expected, name
    full_dependency = {str(path.relative_to(dependency)): digest(path)
                       for path in sorted(dependency.rglob('*')) if path.is_file()}
    (ROOT / 'e878-full-files-pre.json').write_text(json.dumps(full_dependency, indent=2) + '\n')
    for side, role in (('after', 'source'), ('before', 'preopt')):
        record = next(item for item in manifest['sources'] if item['side'] == side)
        original = Path(record['source'])
        head = subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == record['head']
        assert not subprocess.check_output(['git', '-C', str(original), 'status', '--porcelain=v1', '-uno'], text=True)
        tracked = subprocess.check_output(['git', '-C', str(original), 'ls-files', '-z']).decode().strip('\0').split('\0')
        assert set(tracked) == set(record['files_sha256'])
        role_root = ROOT / role
        snapshot = role_root / 'source'
        snapshot.mkdir(parents=True)
        files = {}
        for name in tracked:
            source = original / name
            assert digest(source) == record['files_sha256'][name], name
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files[name] = digest(destination)
        generated = {}
        for kind in ('decode', 'prefill'):
            name = f'flash_attentionXe35_{kind}_64_bf16.cpp'
            source = ARCHIVE / f'build-{side}/src/sycl' / name
            generated[name] = {'original': str(source), 'sha256': digest(source)}
        (role_root / 'source-pre.json').write_text(json.dumps({'role': role, 'head': head, 'original': str(original),
            'snapshot': str(snapshot), 'tracked_files': files, 'original_generated_instances': generated,
            'dependency': str(dependency), 'dependency_historical_label': manifest['cutlass']['historical_revision_label'],
            'original_dependency_record_files': len(manifest['cutlass']['files_sha256']),
            'full_dependency_files': len(full_dependency), 'adaptation': 'only reviewed two-line BuildFlags IntelLLVM gate pending apply_patch'}, indent=2) + '\n')
        print(role, head, 'tracked snapshot files', len(files), 'generated64BF16 instances', len(generated))
    print('e878 exact archived record hashes verified', len(manifest['cutlass']['files_sha256']), 'full current files', len(full_dependency))


if __name__ == '__main__':
    main()

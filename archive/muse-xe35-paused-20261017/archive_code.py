import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


OUTPUT = Path(__file__).resolve().parent
ARTIFACTS = Path('/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017')
SUFFIXES = {'.py', '.sh', '.bash', '.cmake', '.cpp', '.cc', '.c', '.h', '.hpp', '.in', '.diff', '.patch'}


def git(directory, *arguments, environment=None):
    return subprocess.check_output(['git', '-c', f'safe.directory={directory}', '-C', str(directory), *arguments], env=environment)


def excluded(name):
    return (name in {'.git', '__pycache__', '.cache-review', 'deps', '_deps', 'CMakeFiles', 'site-packages',
                     'node_modules', 'results', 'review_logs', 'build', 'cutlass', 'sycl-tla', 'site', 'source', 'runs'}
            or 'venv' in name or 'virtualenv' in name or name.endswith('-build')
            or name.startswith(('build-', 'build_', 'site-', 'install-', 'source-snapshot')))


def is_code(path):
    return path.suffix in SUFFIXES or path.name == 'CMakeLists.txt'


def snapshot_repository(directory):
    head = git(directory, 'rev-parse', 'HEAD').decode().strip()
    branch = git(directory, 'branch', '--show-current').decode().strip()
    role = 'upstream-pr' if directory.parent.name == 'pr' else 'personal-source'
    destination = OUTPUT / 'repositories' / role / directory.name
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='muse-code-archive-index-') as temporary:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git(directory, 'read-tree', head, environment=environment)
        tracked = [os.fsdecode(raw) for raw in git(directory, 'diff', head, '--name-only', '-z').split(b'\x00')
                   if raw and is_code(Path(os.fsdecode(raw)))]
        untracked = []
        for raw in git(directory, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\x00'):
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            path = directory / relative
            if is_code(relative) and not any(excluded(part) for part in relative.parts[:-1]) and not path.is_symlink():
                untracked.append(str(relative))
        for paths, options in ((tracked, ['add', '-u']), (untracked, ['add'])):
            for offset in range(0, len(paths), 50):
                git(directory, *options, '--', *paths[offset:offset + 50], environment=environment)
        patch = git(directory, 'diff', '--cached', '--binary', '--full-index', head, environment=environment)
        (destination / 'working-tree.patch').write_bytes(patch)
        if patch:
            git(directory, 'read-tree', head, environment=environment)
            git(directory, 'apply', '--cached', '--check', str(destination / 'working-tree.patch'), environment=environment)
    metadata = {'directory': str(directory), 'head': head, 'branch': branch,
                'patch_sha256': hashlib.sha256(patch).hexdigest(), 'tracked_paths': tracked,
                'untracked_code': untracked, 'apply_check': 'PASS' if patch else 'NO_CHANGES'}
    (destination / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return metadata


def main():
    repositories = [Path('/home/syk/sgl-kernel-xpu')]
    repositories += sorted(Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr').glob('muse-*-xe35'))
    repositories += sorted(Path('/home/syk/worktrees/sgl-kernel-xpu').glob('muse-spark-*-main-20261014'))
    repositories.append(Path('/home/syk/worktrees/sgl-kernel-xpu/muse-permutation-only-main-20261015'))
    metadata = [snapshot_repository(directory) for directory in repositories if (directory / '.git').exists()]
    files = []
    errors = []
    for directory, directories, names in os.walk(ARTIFACTS, onerror=lambda error: errors.append(str(error))):
        directories[:] = [name for name in directories if not excluded(name)]
        for name in names:
            path = Path(directory) / name
            if not is_code(path) or path.is_symlink():
                continue
            try:
                content = path.read_bytes()
                if b'\x00' in content:
                    continue
                content.decode('utf-8')
                relative = path.relative_to(ARTIFACTS)
                destination = OUTPUT / 'temporary-code' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                files.append({'path': str(relative), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            except (OSError, UnicodeDecodeError) as error:
                errors.append(f'{path}: {error}')
    manifest = {'state': 'PAUSED_NOT_ACCEPTED', 'code_only': True, 'repositories': metadata,
                'temporary_code': files, 'errors': errors, 'raw_data_and_logs_archived': False}
    (OUTPUT / 'CODE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'repositories': len(metadata), 'temporary_code_files': len(files),
                      'temporary_code_bytes': sum(item['bytes'] for item in files), 'errors': errors}, indent=2))


if __name__ == '__main__':
    main()

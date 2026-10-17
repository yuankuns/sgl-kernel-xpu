import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import time


OFFICIAL_BASE = '833c95208dccd2af5628c121222f91186a5b2de7'
EXCLUDED_DIRECTORIES = {'.git', '.venv', 'venv', 'build', 'dist', 'artifact', 'artifacts', '__pycache__', 'CMakeFiles'}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_output(source, *arguments):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(source), *arguments])


def excluded(name):
    parts = pathlib.PurePosixPath(name).parts
    return any(part in EXCLUDED_DIRECTORIES for part in parts[:-1]) or any(part.startswith('build-') or part.startswith('build.') for part in parts[:-1])


def file_record(path):
    attributes = path.lstat()
    if not stat.S_ISREG(attributes.st_mode):
        raise RuntimeError(f'Only regular source files are supported; refusing {path}')
    return {'sha256': digest_file(path), 'size': attributes.st_size, 'mode': stat.S_IMODE(attributes.st_mode)}


def inventory(directory):
    result = {}
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories:
            path = pathlib.Path(parent) / name
            if path.is_symlink():
                raise RuntimeError(f'Symlink directory is not allowed: {path}')
        for name in files:
            path = pathlib.Path(parent) / name
            result[path.relative_to(directory).as_posix()] = file_record(path)
    return result


def verify_inventory(directory, expected):
    actual = inventory(directory)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise RuntimeError(f'Snapshot mismatch: missing={missing}, extra={extra}, changed={changed}')
    return actual


def manifest_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def separate_paths(first, second):
    if first == second or first in second.parents or second in first.parents:
        raise RuntimeError(f'Paths must be disjoint: {first}, {second}')


def source_inventory(source):
    tracked = set(git_output(source, 'ls-files', '-c', '-z').decode().rstrip('\0').split('\0')) - {''}
    untracked = set(git_output(source, 'ls-files', '-o', '--exclude-standard', '-z').decode().rstrip('\0').split('\0')) - {''}
    selected = sorted(name for name in tracked | untracked if not excluded(name))
    missing = [name for name in selected if not (source / name).exists() and not (source / name).is_symlink()]
    files = {}
    for name in selected:
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError(f'Unsafe source path: {name}')
        path = source / name
        if name not in missing:
            if any(parent.is_symlink() for parent in path.parents if parent != source and source in parent.parents):
                raise RuntimeError(f'Symlink ancestor: {path}')
            files[name] = file_record(path)
    return files, {'tracked_count': len(tracked), 'untracked_nonignored_count': len(untracked), 'deleted_tracked': sorted(set(missing) & tracked), 'excluded_paths': sorted(name for name in tracked | untracked if excluded(name))}


def overlay(arguments):
    source = pathlib.Path(arguments.source).resolve(strict=True)
    destination_input = pathlib.Path(arguments.destination).absolute()
    if destination_input.is_symlink():
        raise RuntimeError('Destination must not be a symlink')
    destination = destination_input.resolve()
    manifest = pathlib.Path(arguments.manifest).resolve()
    separate_paths(source, destination)
    for directory in [source, destination]:
        if manifest == directory or directory in manifest.parents:
            raise RuntimeError('Manifest must be outside source and snapshot directories')
    if (destination / '.git').exists() or any((parent / '.git').exists() for parent in destination.parents):
        raise RuntimeError('Refusing to overlay a Git checkout or a directory inside one')
    head = git_output(source, 'rev-parse', 'HEAD').decode().strip()
    branch = git_output(source, 'symbolic-ref', '--quiet', '--short', 'HEAD').decode().strip()
    if head != arguments.expected_head:
        raise RuntimeError(f'Unexpected source HEAD: {head}')
    if arguments.expected_branch and branch != arguments.expected_branch:
        raise RuntimeError(f'Unexpected source branch: {branch}')
    lock = pathlib.Path(arguments.lock_file).resolve() if arguments.lock_file else destination.with_name(destination.name + '.overlay.lock')
    if any(directory == lock or directory in lock.parents for directory in [source, destination]):
        raise RuntimeError('Overlay lock must be outside source and snapshot')
    with exclusive_lock(lock):
        files, selection = source_inventory(source)
        if arguments.verify_only:
            verify_inventory(destination, files)
            print(json.dumps({'verified': True, 'source_digest': manifest_digest(files), 'file_count': len(files)}))
            return
        if manifest.exists():
            raise RuntimeError('Use a new versioned output manifest; refusing to overwrite evidence')
        if destination.exists() and any(destination.iterdir()):
            if not arguments.previous_manifest:
                raise RuntimeError('Nonempty snapshot requires --previous-manifest')
            previous = json.loads(pathlib.Path(arguments.previous_manifest).read_text())
            if pathlib.Path(previous['destination']).resolve() != destination:
                raise RuntimeError('Previous manifest belongs to another destination')
            if previous['source_digest'] != manifest_digest(previous['files']):
                raise RuntimeError('Previous source manifest digest is invalid')
            verify_inventory(destination, previous['files'])
        destination.mkdir(parents=True, exist_ok=True)
        old = inventory(destination)
        removed = sorted(set(old) - set(files))
        for name in removed:
            (destination / name).unlink()
        for parent, directories, filenames in os.walk(destination, topdown=False):
            directory = pathlib.Path(parent)
            if directory != destination and not any(directory.iterdir()):
                directory.rmdir()
        copied = []
        preserved = []
        for name, record in files.items():
            target = destination / name
            origin = source / name
            if old.get(name, {}).get('sha256') == record['sha256'] and target.is_file():
                before = target.stat().st_mtime_ns
                os.chmod(target, record['mode'])
                if target.stat().st_mtime_ns != before:
                    raise RuntimeError(f'Identical file mtime unexpectedly changed: {name}')
                preserved.append(name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(dir=target.parent)
                os.close(descriptor)
                temporary = pathlib.Path(temporary_name)
                try:
                    shutil.copyfile(origin, temporary)
                    os.chmod(temporary, record['mode'])
                    if file_record(temporary) != record:
                        raise RuntimeError(f'Source changed during copy: {name}')
                    os.replace(temporary, target)
                    copied.append(name)
                finally:
                    temporary.unlink(missing_ok=True)
        verify_inventory(destination, files)
        current_files, current_selection = source_inventory(source)
        if current_files != files or current_selection != selection or git_output(source, 'rev-parse', 'HEAD').decode().strip() != head or git_output(source, 'symbolic-ref', '--quiet', '--short', 'HEAD').decode().strip() != branch:
            raise RuntimeError('Read-only source changed during overlay; snapshot must not be built')
        payload = {'schema': 1, 'created_epoch': time.time(), 'source': str(source), 'destination': str(destination), 'official_base': OFFICIAL_BASE, 'head': head, 'branch': branch, 'source_digest': manifest_digest(files), 'files': files, 'selection': selection, 'copied': copied, 'mtime_preserved': preserved, 'removed': removed, 'verified': True, 'previous_manifest': arguments.previous_manifest}
        write_json(manifest, payload)
        print(json.dumps({'manifest': str(manifest), 'source_digest': payload['source_digest'], 'files': len(files), 'copied': len(copied), 'mtime_preserved': len(preserved), 'removed': len(removed)}))


def main():
    parser = argparse.ArgumentParser(description='CPU-only byte-verified canonical source overlay; never configure or build')
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--previous-manifest')
    parser.add_argument('--expected-head', default=OFFICIAL_BASE)
    parser.add_argument('--expected-branch')
    parser.add_argument('--lock-file')
    parser.add_argument('--verify-only', action='store_true')
    overlay(parser.parse_args())


if __name__ == '__main__':
    main()

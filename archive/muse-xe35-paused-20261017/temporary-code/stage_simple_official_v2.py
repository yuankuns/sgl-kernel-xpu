import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    source, build, artifact = map(Path, sys.argv[1:])
    command = ['git', '-c', f'safe.directory={source}', '-C', str(source)]
    head = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip()
    if head != '833c95208dccd2af5628c121222f91186a5b2de7':
        raise RuntimeError(f'Unexpected upstream base: {head}')
    files = subprocess.check_output(command + ['ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    stage = artifact / 'after/site'
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / 'python/sgl_kernel', stage / 'sgl_kernel', dirs_exist_ok=True)
    libraries = {library.name: library for library in (build / 'src').glob('*.so')}
    for library in (build / 'src').rglob('common_ops*.so'):
        if library.name in libraries and libraries[library.name] != library:
            raise RuntimeError(f'Ambiguous registration library: {library}')
        libraries[library.name] = library
    if not any(name.startswith('common_ops') for name in libraries):
        raise RuntimeError('Full build did not produce real common_ops registration')
    for name, library in libraries.items():
        shutil.copy2(library, stage / 'sgl_kernel' / name)
    manifest = {
        'head': head, 'branch': subprocess.check_output(command + ['branch', '--show-current'], text=True).strip(),
        'source': str(source), 'build': str(build),
        'stage_method': 'full default official build libraries plus unmodified official port Python package; no ABI diagnostic bridge',
        'files': {filename: hashlib.sha256((source / filename).read_bytes()).hexdigest() for filename in files if filename and (source / filename).is_file()},
        'build_libraries': {name: hashlib.sha256(library.read_bytes()).hexdigest() for name, library in sorted(libraries.items())},
        'stage': {str(filename.relative_to(stage)): hashlib.sha256(filename.read_bytes()).hexdigest() for filename in stage.rglob('*') if filename.is_file()},
    }
    (artifact / 'source-binary-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'source': str(source), 'head': head, 'native_library_count': len(libraries), 'manifest': str(artifact / 'source-binary-manifest.json')}))


if __name__ == '__main__':
    main()

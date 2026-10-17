import hashlib
import json
import pathlib
import subprocess

artifact = pathlib.Path(__file__).resolve().parent
target = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35')
source = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
dependency = artifact.parent / 'deps/sycl-tla-7b56'


def command_xe35(arguments):
    result = subprocess.run(arguments, text=True, capture_output=True)
    return {'command': arguments, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def hash_file_xe35(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


records = {'repositories': [], 'source_hashes': [], 'binary_hashes': [], 'exits': {}}
for repository in (source, target, dependency):
    for arguments in (['rev-parse', 'HEAD'], ['status', '--short', '--branch'], ['remote', '-v']):
        records['repositories'].append(command_xe35(['git', '-C', str(repository), *arguments]))
records['source_clean_against_revision'] = command_xe35(['git', '-C', str(source), 'diff', '--exit-code', 'd983e87', '--', 'src', 'include', 'python', 'tests', 'benchmark'])
for repository in (source, target):
    paths = list(repository.glob('src/sycl/xe35/*nvfp4*'))
    paths += list(repository.glob('src/sycl/xe35/*subbyte*'))
    paths += list(repository.glob('src/sycl/xe35/nvfp4/**/*'))
    paths += list(repository.glob('include/sgl_kernel/xe35/*nvfp4*'))
    paths += [repository / path for path in ('src/BuildOnLinux.cmake', 'src/CMakeLists.txt', 'src/torch_extension_sycl.cc', 'include/sgl_kernel_ops.h', 'python/sgl_kernel/__init__.py')]
    paths += [repository / path for path in ('src/sycl/nvfp4_woq.cpp', 'src/sycl/mxfp4_moe_scale_layout.cpp')]
    paths += list(repository.glob('python/sgl_kernel/*nvfp4*'))
    paths += list(repository.glob('tests/*nvfp4*'))
    paths += list(repository.glob('benchmark/*nvfp4*'))
    for path in sorted(set(paths)):
        if path.is_file():
            records['source_hashes'].append(hash_file_xe35(path))
for path in sorted(source.glob('build/src/*.so')):
    records['binary_hashes'].append(hash_file_xe35(path))
    records.setdefault('source_elf_dependencies', []).append(command_xe35(['readelf', '-d', str(path)]))
for directory in ('target-wheels', 'baseline-wheels', 'target-default-build/src'):
    for path in sorted((artifact / directory).glob('*')):
        if path.is_file() and path.suffix in ('.whl', '.so'):
            records['binary_hashes'].append(hash_file_xe35(path))
for path in sorted(artifact.glob('*.exit')):
    records['exits'][path.name] = path.read_text().strip()
print(json.dumps(records, indent=2))

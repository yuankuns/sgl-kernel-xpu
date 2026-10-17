import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['source', 'port'])
    args = parser.parse_args()
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('CPU-only selector and CPUs96–111 required')
    folder = Path(__file__).resolve().parent
    root = folder.parent
    expected = root / ('fresh-source-icpx-v1' if args.role == 'source' else 'canonical-v4-icpx') / 'site/sgl_kernel'
    import torch
    import sgl_kernel
    import torch.xpu.graphs
    assert Path(sgl_kernel.__file__).resolve() == expected / '__init__.py'
    maps_text = Path('/proc/self/maps').read_text()
    (folder / (args.role + '-actual-CPU-maps.txt')).write_text(maps_text)
    paths = sorted({line.split()[-1] for line in maps_text.splitlines() if line.split()[-1].startswith('/')})
    mapped = {name: {'sha256': digest(Path(name)), 'size': Path(name).stat().st_size} for name in paths if Path(name).is_file()}
    package_maps = {name: record for name, record in mapped.items() if str(expected) in name}
    assert len(package_maps) == (156 if args.role == 'source' else 712)
    graphs = Path(inspect.getsourcefile(torch.xpu.graphs))
    header = Path(torch.__file__).parent / 'include/ATen/xpu/XPUGraph.h'
    for path in [graphs, header]:
        shutil.copyfile(path, folder / (args.role + '-' + path.name))
    torch_library = Path(torch.__file__).parent / 'lib/libtorch_xpu.so'
    commands = {
        'xpu-symbols': ['nm', '-D', '-C', str(torch_library)],
        'xpu-needed': ['readelf', '-d', str(torch_library)],
        'xpu-buildid': ['readelf', '-n', str(torch_library)],
        'sycl-version': ['/opt/gfx-deps/dpcpp/bin/icpx', '--version'],
    }
    exits = {}
    for name, command in commands.items():
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (folder / (args.role + '-' + name + '.log')).write_text(completed.stdout)
        exits[name] = completed.returncode
    record = {'epoch': time.time(), 'role': args.role, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'torch_version': str(torch.__version__), 'torch_git_version': torch.version.git_version,
              'torch_config': torch.__config__.show(), 'python': sys.version, 'sgl_package': str(expected),
              'actual_mapped_files': mapped, 'normal_package_mapped_count': len(package_maps),
              'torch_graphs_sha256': digest(graphs), 'torch_XPUGraph_header_sha256': digest(header),
              'torch_xpu_library': str(torch_library), 'torch_xpu_library_sha256': digest(torch_library),
              'inspection_command_exits': exits, 'GPU_initialized_by_this_script': False,
              'no_XPU_is_available_query_or_graph_or_event_or_replay_called': True,
              'provenance_scope': 'Current actual CPU normalimport mappings; not claimed recovered v6 failed GPU process maps'}
    (folder / (args.role + '-CPU-runtime-provenance.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'role': args.role, 'native_maps': len(package_maps), 'all_mapped_files': len(mapped), 'torch_version': str(torch.__version__), 'GPU_work': False}), flush=True)


if __name__ == '__main__':
    main()

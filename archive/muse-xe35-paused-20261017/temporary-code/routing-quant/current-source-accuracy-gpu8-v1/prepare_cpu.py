import hashlib
import json
import os
from pathlib import Path
import subprocess

import sgl_kernel
import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    stage = artifact / 'fresh-current-source-icpx-native-v3'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert not torch.xpu.is_initialized()
    assert Path(sgl_kernel.__file__).resolve() == stage / 'site/sgl_kernel/__init__.py'
    assert int((stage / 'cpu-native-acceptance.exit').read_text()) == 0
    acceptance = json.loads((stage / 'cpu-native-acceptance.json').read_text())
    assert acceptance['fresh_TripleOps_BF16_accumulator_actual_type'] == 'float'
    maps = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line}
    assert maps == acceptance['current_mapped_native_SHA256'].keys()
    for name, expected in acceptance['current_mapped_native_SHA256'].items():
        assert digest(Path(name)) == expected
    source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    protected = json.loads((stage / 'source-preverify.json').read_text())
    for name, expected in protected['original_tracked'].items():
        assert digest(source / name) == expected
    groups = {'original-silu-and-mul-opmath-14': ['test_silu_and_mul_opmath.py'],
              'original-full-1300': ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4.py', 'test_topk_sigmoid.py', 'test_routing_benchmark_scope.py']}
    collected = {}
    for name, files in groups.items():
        command = ['/opt/gfx-deps/venv/bin/python', '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider',
                   *[str(source / 'tests' / filename) for filename in files]]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / f'CPU-collect-{name}.log').write_text(result.stdout)
        (directory / f'CPU-collect-{name}.exit').write_text(str(result.returncode) + '\n')
        assert result.returncode == 0, result.stdout
        nodes = [line for line in result.stdout.splitlines() if '::' in line and not line.startswith(' ')]
        assert len(nodes) == (14 if name.endswith('-14') else 1300), len(nodes)
        (directory / f'CPU-collected-{name}-nodeids.json').write_text(json.dumps(nodes, indent=2) + '\n')
        collected[name] = {'count': len(nodes), 'original_files_SHA256': {filename: digest(source / 'tests' / filename) for filename in files}}
    checked = {'package': sgl_kernel.__file__, 'mapped_native_sha256': acceptance['current_mapped_native_SHA256'],
               'Torch_provider': torch.__file__, 'Torch_version': torch.__version__,
               'GPU_initialized': False, 'fresh_tested_native_SHA256': acceptance['four_tested_native_SHA256']}
    (directory / 'normal-source-precheck.json').write_text(json.dumps(checked, indent=2) + '\n')
    assert not torch.xpu.is_initialized()
    report = {'CPU_preparation_exit': 0, 'collection': collected, 'normal_SOURCE_package_and_four_fresh_maps_verified': True,
              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
              'current_SOURCE_source_native_site_manifest': str(stage / 'source-native-package-manifest.json'),
              'GPU_work_started': False, 'GPU_grant_issued': False, 'no_old_legacy_package_runtime': True}
    (directory / 'CPU-preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

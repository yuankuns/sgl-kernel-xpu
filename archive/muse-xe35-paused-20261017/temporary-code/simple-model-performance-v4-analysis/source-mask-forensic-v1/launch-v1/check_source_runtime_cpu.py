import hashlib
import json
import os
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
FORENSIC = DIRECTORY.parent
ROOT = FORENSIC.parents[1]


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    assert set(os.sched_getaffinity(0)) == set(range(80, 96))
    assert os.environ['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:cpu'
    comparator_path = ROOT / 'logits-vocab/canonical-validation-v4-icpx/source-comparator-manifest.json'
    comparator = json.loads(comparator_path.read_text())
    assert comparator['source_revision'] == '448f0c0cf6cd13de7c2d6ec28de1d392039fdaca'
    native_manifest = Path(comparator['source_native_manifest'])
    assert digest(native_manifest) == comparator['source_native_manifest_sha256']
    stage = Path(comparator['stage'])
    expected = {name: record['sha256'] for name, record in comparator['files'].items()}
    actual = {str(path.relative_to(stage)): digest(path) for path in stage.rglob('*') if path.is_file()}
    assert actual == expected
    import torch
    import sgl_kernel

    assert not torch.xpu.is_initialized()
    torch_root = Path('/opt/gfx-deps/venv/lib/python3.12/site-packages/torch')
    assert Path(torch.__file__).resolve().parent == torch_root
    assert Path(sys.executable).absolute() == Path('/opt/gfx-deps/venv/bin/python')
    assert Path(sgl_kernel.__file__).resolve() == stage / 'sgl_kernel/__init__.py'
    assert Path(sgl_kernel.common_ops.__file__).resolve() == stage / comparator['registration']
    assert {str(path.relative_to(stage)): digest(path) for path in stage.rglob('*') if path.is_file()} == expected
    mapped = {}
    torch_mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(('libsgl-ops-', 'common_ops.')):
            assert path.is_relative_to(stage), 'Mixed SOURCE package native mapping'
            relative = path.relative_to(stage).as_posix()
            assert relative in expected and digest(path) == expected[relative]
            mapped[relative] = expected[relative]
        elif path.name.startswith(('libtorch', 'libc10')):
            assert path.parent == torch_root / 'lib', 'Foreign Torch native provider'
            torch_mapped[str(path)] = digest(path)
    old_gate = json.loads((ROOT / 'simple-model-performance-v4-cpu-preparation/logits-vocab-before.json').read_text())
    assert mapped == old_gate['actual_mapped_native_libraries']
    tested = 'sgl_kernel/libsgl-ops-sycl-VocabMask.so'
    assert mapped[tested] == '443db8771ef6cf36395f061cf1045ad3d83fb5c29c01367a65f087289853ee4a'
    schema = str(torch.ops.sgl_kernel.vocab_mask_xe35.default._schema)
    assert not torch.xpu.is_initialized()
    record = {'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'Torch_XPU_initialized': False, 'GPU_started': False, 'compiler_started': False, 'python': sys.executable, 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__, 'Torch_native_mappings_SHA256': torch_mapped, 'normal_stage': str(stage), 'normal_stage_files': actual, 'normal_stage_tree_SHA256': hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'actual_native_mappings': mapped, 'native_mapped_count': len(mapped), 'SOURCE_native_sha256': mapped[tested], 'comparator_sha256': digest(comparator_path), 'source_native_manifest_sha256': digest(native_manifest), 'schema': schema, 'input_recipe': {'ids': [-1, 0, 1, 202047, 202048, 100, 70000, 150000], 'dtype': 'int64', 'org_start': 0, 'org_end': 202048, 'num_org_padding': 0, 'added_start': 202048, 'added_end': 202048, 'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}}
    with (DIRECTORY / 'RUNTIME_SOURCE_CPU_PROVIDER.json').open('x') as output:
        output.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'SOURCE_stage_files': len(actual), 'SOURCE_native_maps': len(mapped), 'Torch_XPU_initialized': False}))


if __name__ == '__main__':
    main()

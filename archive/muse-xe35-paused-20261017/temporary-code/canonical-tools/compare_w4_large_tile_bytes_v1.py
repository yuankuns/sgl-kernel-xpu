import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / 'mxfp4-w4a4/aot-regression-main-cpu-v1'


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    comparison = json.loads((ROOT / 'METADATA_COMPARISON_v3.json').read_text())
    selected = [row for row in comparison['rows'] if row['pipeline_and_tile'] == '4-256-256-128']
    assert len(selected) == 1
    binaries = {}
    for role in ('source', 'after'):
        metadata = json.loads((ROOT / role / 'metadata.json').read_text())
        matches = [kernel for image in metadata['images'] for kernel in image['kernels']
                   if kernel['ISA_sha256'] == selected[0][role]['ISA_sha256']]
        assert len(matches) == 1
        binaries[role] = Path(matches[0]['ISA_file']).read_bytes()
        assert hashlib.sha256(binaries[role]).hexdigest() == selected[0][role]['ISA_sha256']
    assert len(binaries['source']) == len(binaries['after'])
    offsets = [index for index, (before, after) in enumerate(zip(binaries['source'], binaries['after'])) if before != after]
    prefix = min(offsets) if offsets else len(binaries['source'])
    assert binaries['source'][:prefix] == binaries['after'][:prefix]
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_or_compiler_started': False,
              'pipeline_and_tile': '4-256-256-128', 'ISA_bytes': len(binaries['source']),
              'differing_byte_offsets': offsets, 'equal_prefix_bytes': prefix,
              'equal_prefix_sha256': hashlib.sha256(binaries['source'][:prefix]).hexdigest(),
              'decoder_exit_1_retained': True, 'EOT_or_instruction_semantics_proven': False,
              'runtime_grid_or_performance_root_cause_proven': False}
    with (ROOT / 'LARGE_TILE_BYTE_COMPARISON_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

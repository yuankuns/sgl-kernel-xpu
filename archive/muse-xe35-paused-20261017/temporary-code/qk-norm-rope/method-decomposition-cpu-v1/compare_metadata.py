import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
REWRITES = {'26FusedQKNormRopeKernel_xe35': '21FusedQKNormRopeKernel',
    '24qk_norm_rope_kernel_xe3511Kernel_xe35': '23fused_qk_norm_rope_xe356Kernel',
    'at::native::xpu::qk_norm_rope_kernel_xe35::LaunchConfig_xe35': 'at::native::xpu::fused_qk_norm_rope_xe35::LaunchConfig'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    for before, after in REWRITES.items():
        text = text.replace(before, after)
    return text


def disassemble(path, log):
    arguments = ['/usr/local/bin/iga64', '-p=3pxpc', '-d', '-Xprint-pc', str(path)]
    result = subprocess.run(arguments, capture_output=True, text=True)
    log.with_suffix('.command.json').write_text(json.dumps(arguments) + '\n')
    log.with_suffix('.exit').write_text(f'{result.returncode}\n')
    log.with_suffix('.stderr').write_text(result.stderr)
    log.write_text(result.stdout)
    assert result.returncode == 0
    return result.stdout


def executable_end(text):
    instructions = []
    for line in text.splitlines():
        match = re.search(r'\[([0-9A-Fa-f]+)\]', line)
        if match:
            instructions.append((int(match.group(1), 16), line))
    last_eot = max(index for index, item in enumerate(instructions) if '{EOT' in item[1])
    assert last_eot + 1 < len(instructions)
    return instructions[last_eot + 1][0]


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    source = json.loads((RUN / 'source/metadata.json').read_text())
    port = json.loads((RUN / 'port/metadata.json').read_text())
    rows = []
    image_rows = []
    for before, after in zip(source['images'], port['images']):
        index = before['index']
        assert index == after['index'] and len(before['kernels']) == len(after['kernels'])
        original_metadata = (RUN / 'source' / f'image-{index:02d}/ze_info.yaml').read_text()
        port_metadata = (RUN / 'port' / f'image-{index:02d}/ze_info.yaml').read_text()
        name_only_normalized = normalized(port_metadata)
        image_rows.append({'image': index, 'SOURCE_metadata_sha256': sha(original_metadata.encode()),
            'PORT_raw_metadata_sha256': sha(port_metadata.encode()),
            'PORT_exact_three_name_rewrites_metadata_sha256': sha(name_only_normalized.encode()),
            'metadata_equal_after_only_explicit_symbol_type_name_rewrites': original_metadata == name_only_normalized})
        assert original_metadata == name_only_normalized
        for kernel_index, (left, right) in enumerate(zip(before['kernels'], after['kernels'])):
            assert left['section_name'] == normalized(right['section_name']) and left['bytes'] == right['bytes']
            paths = {role: RUN / role / f'image-{index:02d}/kernel-{kernel_index:03d}.isa.bin' for role in ('source', 'port')}
            binaries = {role: path.read_bytes() for role, path in paths.items()}
            texts = {role: disassemble(path, path.with_suffix('.iga-full.txt')) for role, path in paths.items()}
            end = executable_end(texts['source'])
            assert end == executable_end(texts['port'])
            differences = [offset for offset, (original, copied) in enumerate(zip(binaries['source'], binaries['port'])) if original != copied]
            assert differences and all(offset >= end for offset in differences)
            assert binaries['source'][:end] == binaries['port'][:end]
            changed_lines = [(left_line, right_line) for left_line, right_line in zip(texts['source'].splitlines(), texts['port'].splitlines()) if left_line != right_line]
            assert len(changed_lines) == 2 and all('mov (16|M0)' in line and 'null<1>:ud' in line for pair in changed_lines for line in pair)
            rows.append({'image': index, 'kernel': kernel_index, 'SOURCE_name': left['section_name'], 'PORT_name': right['section_name'],
                'length_bytes': left['bytes'], 'executable_end_after_last_EOT_offset': end,
                'SOURCE_full_ISA_sha256': sha(binaries['source']), 'PORT_full_ISA_sha256': sha(binaries['port']),
                'SOURCE_executable_prefix_sha256': sha(binaries['source'][:end]),
                'PORT_executable_prefix_sha256': sha(binaries['port'][:end]), 'raw_differing_byte_offsets': differences,
                'differences_only_two_null_MOV_immediates_after_final_EOT': True,
                'disassembly_changed_lines': changed_lines})
    assert len(rows) == 80
    result = {'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'kernel_pairs': len(rows),
        'images': image_rows, 'explicit_metadata_name_rewrites': REWRITES, 'rows': rows,
        'all_executable_prefixes_through_last_EOT_byte_identical': True,
        'raw_full_ISA_identical': False, 'differences_only_two_null_MOV_immediates_after_final_EOT': True,
        'IGA_tool_sha256': sha(Path('/usr/local/bin/iga64').read_bytes()), 'IGA_decode_platform': '3pxpc',
        'GPU_work_or_compilation': False, 'host_path_launch_grid_runtime_performance_equivalence_inferred': False}
    (RUN / 'metadata-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'images')}, indent=2))


if __name__ == '__main__':
    main()

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / 'mxfp4-w4a4/aot-regression-main-cpu-v1'


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    comparison = json.loads((ROOT / 'METADATA_COMPARISON_v3.json').read_text())
    selected = [row for row in comparison['rows'] if row['pipeline_and_tile'] == '4-256-256-128']
    assert len(selected) == 1
    evidence = {}
    binaries = {}
    directory = ROOT / 'large-tile-disassembly-v1'
    directory.mkdir(exist_ok=False)
    for role in ('source', 'after'):
        summary = json.loads((ROOT / role / 'metadata.json').read_text())
        matches = [kernel for image in summary['images'] for kernel in image['kernels']
                   if kernel['ISA_sha256'] == selected[0][role]['ISA_sha256']]
        assert len(matches) == 1
        path = Path(matches[0]['ISA_file'])
        binaries[role] = path.read_bytes()
        arguments = ['/usr/local/bin/iga64', '-p=3pxpc', '-d', '-Xprint-pc', str(path)]
        result = subprocess.run(arguments, capture_output=True, text=True)
        (directory / (role + '.command.json')).write_text(json.dumps(arguments) + '\n')
        (directory / (role + '.stderr')).write_text(result.stderr)
        (directory / (role + '.exit')).write_text(str(result.returncode) + '\n')
        (directory / (role + '.iga.txt')).write_text(result.stdout)
        assert result.returncode == 0
        instructions = []
        for line in result.stdout.splitlines():
            match = re.search(r'\[([0-9A-Fa-f]+)\]', line)
            if match:
                instructions.append((int(match.group(1), 16), line))
        eot = [index for index, instruction in enumerate(instructions) if '{EOT' in instruction[1]]
        assert eot
        last = max(eot)
        end = instructions[last + 1][0] if last + 1 < len(instructions) else len(binaries[role])
        evidence[role] = {'ISA_sha256': matches[0]['ISA_sha256'], 'instruction_lines': len(instructions),
                          'last_EOT_end_offset': end,
                          'executable_prefix_sha256': hashlib.sha256(binaries[role][:end]).hexdigest(),
                          'disassembly_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()}
    same_end = evidence['source']['last_EOT_end_offset'] == evidence['after']['last_EOT_end_offset']
    same_prefix = same_end and evidence['source']['executable_prefix_sha256'] == evidence['after']['executable_prefix_sha256']
    differences = [offset for offset, (before, after) in enumerate(zip(binaries['source'], binaries['after'])) if before != after]
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_or_compiler_started': False,
              'pipeline_and_tile': '4-256-256-128', 'role_evidence': evidence,
              'executable_prefix_through_last_EOT_byte_identical': same_prefix,
              'all_differing_byte_offsets': differences,
              'runtime_instance_grid_or_performance_root_cause_proven': False}
    with (directory / 'SUMMARY.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'all_differing_byte_offsets'}))


if __name__ == '__main__':
    main()

import hashlib
import html
import json
from pathlib import Path
import re


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
PERFORMANCE = ARTIFACT / 'official-performance-v4-gpu6-v2'
SHAPES = ('262144-4096-4096', '262144-8192-4096')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def normal(kind):
    import sgl_kernel
    import torch

    check = json.loads((PERFORMANCE / f'CPU-normal-{kind}.json').read_text())
    package = Path(sgl_kernel.__file__).resolve()
    expected = Path(check['package'])
    assert package == expected or package.parent == expected
    maps = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line}
    inventory = check.get('mapped_native_sha256', check.get('mapped_native_hashes'))
    assert maps == set(inventory), 'normal current mapped native set changed'
    for name, expected_hash in inventory.items():
        assert digest(name) == expected_hash, name
    site = Path(check.get('site', str(package.parent.parent)))
    files = check.get('package_files_sha256', check.get('package_files_SHA256'))
    if files:
        for name, expected_hash in files.items():
            assert digest(site / name) == expected_hash, name
    return {'package': str(package), 'Torch_provider': torch.__file__, 'Torch_version': torch.__version__,
            'mapped_native_sha256': inventory, 'mapped_native_count': len(maps)}


def dot_launches(text, kind, calls):
    metadata = json.loads((ARTIFACT / 'aot-regression-main-cpu-v1/METADATA_COMPARISON_v3.json').read_text())
    known = {row[kind]['name']: row for row in metadata['rows']}
    rows = []
    for table in re.findall(r'<TABLE\b[^>]*>.*?</TABLE>', text, re.S):
        fields = {}
        for key, value in re.findall(r'<TR><TD[^>]*>([^<]+)</TD><TD[^>]*>(.*?)</TD></TR>', table, re.S):
            fields[html.unescape(key)] = html.unescape(re.sub('<[^>]+>', '', value))
        name = fields.get('kernelName', '')
        if 'GemmUniversal' not in name:
            continue
        assert name in known, 'actual GEMM name absent from already audited seven AOT pairs'
        row = known[name]
        grid = []
        for key in ('groupCounts', 'groupSizes'):
            assert re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*', fields.get(key, '')), f'missing actual three-dimensional {key}'
            grid.append([int(value) for value in fields[key].split('x')])
        assert re.fullmatch(r'0x[0-9a-fA-F]+', fields.get('hKernel', ''))
        rows.append({'kernel_name': name, 'kernel_handle': fields['hKernel'], 'pipeline_and_tile': row['pipeline_and_tile'],
                     'group_counts': grid[0], 'local_size': grid[1],
                     'global_size': [count * size for count, size in zip(*grid)],
                     'AOT_ISA_SHA256': row[kind]['ISA_sha256'], 'all_observed_fields': fields})
    assert len(rows) == calls, f'actual GEMM append count {len(rows)} != captured native calls {calls}'
    assert len({(row['pipeline_and_tile'], tuple(row['group_counts']), tuple(row['local_size'])) for row in rows}) == 1
    return rows


def ur_direct(text, shape, kind):
    handles = {}
    for line in text.splitlines():
        if 'urKernelCreate(' not in line or '<---' not in line:
            continue
        name = re.search(r'\.pKernelName\s*=\s*0x[0-9a-fA-F]+\s*\(([^)]+)\)', line)
        handle = re.search(r'\.phKernel\s*=\s*0x[0-9a-fA-F]+\s*\((0x[0-9a-fA-F]+)\)', line)
        program = re.search(r'\.hProgram\s*=\s*(0x[0-9a-fA-F]+)', line)
        if name and handle and program:
            handles[handle.group(1)] = {'name': name.group(1), 'UR_program_handle': program.group(1)}
    begin = f'W4_DIAG_BEGIN {kind} {shape} DIRECT'
    end = f'W4_DIAG_END {kind} {shape} DIRECT'
    assert text.count(begin) == text.count(end) == 1, 'direct submission marker missing/duplicated'
    window = text.split(begin, 1)[1].split(end, 1)[0]
    launches = []
    for line in window.splitlines():
        if '<---' not in line or not re.search(r'urEnqueueKernelLaunch(?:WithArgsExp)?\(', line):
            continue
        handle = re.search(r'\.hKernel\s*=\s*(0x[0-9a-fA-F]+)', line)
        assert handle and handle.group(1) in handles, 'actual direct kernel handle/name relation unavailable'
        identity = handles[handle.group(1)]
        if 'GemmUniversal' not in identity['name']:
            continue
        assert 'UR_RESULT_SUCCESS' in line, 'direct UR launch not reported success'
        dimensions = {}
        for field in ('pGlobalWorkSize', 'pLocalWorkSize'):
            matched = re.search(r'\.' + field + r'\s*=\s*0x[0-9a-fA-F]+\s*\(([0-9]+)\)', line)
            assert matched and int(matched.group(1)) > 0, 'actual first dimension unavailable'
            dimensions[field + '_first_element_only'] = int(matched.group(1))
        launches.append(identity | dimensions | {'UR_kernel_handle': handle.group(1), 'raw_completed_call': line})
    assert len(launches) == 1, f'actual direct GEMM launch count {len(launches)} != 1'
    return launches[0]

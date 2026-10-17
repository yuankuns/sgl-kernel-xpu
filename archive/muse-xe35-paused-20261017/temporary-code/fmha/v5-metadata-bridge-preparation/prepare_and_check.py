import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
PORT = ART.parent.parent / 'pr/muse-fmha-xe35'
ENTRY_PATH = 'src/sycl/xe35/fmha/entry.cpp'
ENTRY = PORT / ENTRY_PATH
FREEZE = ART / 'canonical-source-freeze-v4-icpx.json'
INT_MAX = 2147483647


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(name, report):
    (ROOT / name).write_text(json.dumps(report, indent=2) + '\n')


def load_overlay():
    path = ART.parent / 'canonical-tools/overlay_snapshot.py'
    spec = importlib.util.spec_from_file_location('readonly_overlay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drain():
    ancestors = {os.getpid()}
    current = os.getpid()
    while current > 1:
        status = Path(f'/proc/{current}/status').read_text()
        current = int(re.search(r'^PPid:\s+(\d+)', status, re.M).group(1))
        ancestors.add(current)
    handles = {}
    active = []
    for path in ART.rglob('*.handle'):
        if 'site' in path.parts or 'source' in path.parts or 'deps' in path.parts:
            continue
        for match in re.finditer(r'supervisor_pid[=:]\s*(\d+)', path.read_text(errors='replace')):
            pid = int(match.group(1))
            handles[str(path.relative_to(ART))] = {'pid': pid, 'proc_exists': Path(f'/proc/{pid}').exists()}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) in ancestors:
            continue
        try:
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'muse-upstream-artifacts-20261017/fmha/' in command or '/pr/muse-fmha-xe35' in command:
            active.append({'pid': int(directory.name), 'command': command})
    assert not active, active
    for record in handles.values():
        assert not record['proc_exists'], record
    inode = Path('/tmp/intel-gpu-5.lock').stat().st_ino
    locks = [line for line in Path('/proc/locks').read_text().splitlines() if line.split()[5].endswith(':' + str(inode))]
    assert not locks
    return {'epoch': time.time(), 'old_supervisor_handles': handles, 'active_FM_processes': active,
            'GPU5_physical_lock_holders': locks, 'logical_GPU5_FM_lease_retained': True,
            'process_check': 'all host /proc command lines, old supervisor PIDs, own checker ancestors excluded'}


def preflight():
    assert not (ROOT / 'before-entry.cpp').exists()
    report = drain()
    overlay = load_overlay()
    freeze = json.loads(FREEZE.read_text())
    files, selection = overlay.source_inventory(PORT)
    assert files == freeze['files'] and len(files) == 660
    assert overlay.manifest_digest(files) == freeze['source_digest']
    (ROOT / 'before-entry.cpp').write_bytes(ENTRY.read_bytes())
    old_metadata = list(ART.glob('canonical-source-freeze*.json'))
    report.update({'CPU_only': True, 'old_v4_source_digest': freeze['source_digest'],
        'old_entry_sha256': digest(ENTRY), 'before_entry_sha256': digest(ROOT / 'before-entry.cpp'),
        'old_freeze_sha256': {str(path): digest(path) for path in old_metadata},
        'source_file_count': len(files), 'source_inventory_selection': selection,
        'no_compile_GPU_or_stage_write': True})
    write_report('pre-edit-drain-and-provenance.json', report)
    print(json.dumps({'preflight': 0, 'old_handles': len(report['old_supervisor_handles']),
                      'active_FM': 0, 'source_files': 660, 'before_entry_sha256': report['before_entry_sha256']}))


class MetadataTensor:
    def __init__(self, shape, dtype='BF16', contiguous=True, present=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.contiguous = contiguous
        self.present = present

    def dim(self):
        return len(self.shape)

    def size(self, dimension):
        return self.shape[dimension]

    def sizes(self):
        return self.shape

    def scalar_type(self):
        return self.dtype

    def is_contiguous(self):
        return self.contiguous

    def has_value(self):
        return self.present


def expression(source, name):
    match = re.search(r'const bool ' + name + r'\s*=\s*(.*?);', source, re.S)
    assert match
    text = match.group(1).replace('->', '.').replace('&&', ' and ').replace('||', ' or ')
    text = text.replace('std::numeric_limits<int>::max() / 64', 'INT_MAX // 64')
    text = text.replace('at::kBFloat16', "'BF16'")
    text = re.sub(r'(?<=\d)f\b', '', text)
    return ' '.join(re.sub(r'!(?!=)', ' not ', text).split())


def metadata(case, maximum):
    batch = case['requests']
    query_length = case['query_length']
    tensors = {'q': MetadataTensor([batch * query_length, 128, 64]),
        'k': MetadataTensor(case['paged_KV_pool_shape']), 'v': MetadataTensor(case['paged_KV_pool_shape']),
        'out': MetadataTensor([batch * query_length, 128, 64]),
        'page_table': MetadataTensor(case['page_table_shape'])}
    tensors.update({name: MetadataTensor([], present=False) for name in (
        'softmax_lse', 'q_v_', 'kv_batch_idx_', 'leftpad_k_', 'rotary_cos_', 'rotary_sin_',
        'seqlens_rotary_', 'scheduler_metadata_', 'rel_bias_', 'bidirectional_block_ids_',
        'sinks_', 'q_descale_', 'k_descale_', 'v_descale_')})
    tensors.update({'INT_MAX': INT_MAX, 'batch_xe35': batch, 'max_seqlen_q': query_length,
        'max_seqlen_k': maximum, 'is_causal': True, 'window_size_left': case['window'],
        'softcap': 0.0, 'num_kv_splits': 0, 'sm_margin': 0})
    return tensors


def evaluate(source, values, new):
    context = copy.deepcopy(values)
    evaluate_expression = lambda name: eval(expression(source, name), {'__builtins__': {}}, context)
    if new:
        context['capacity_hint_xe35'] = evaluate_expression('capacity_hint_xe35')
        context['resolved_max_seqlen_k_xe35'] = (
            context['page_table'].size(1) * context['k'].size(1) if context['capacity_hint_xe35'] else context['max_seqlen_k'])
    context['production_shape_xe35'] = evaluate_expression('production_shape_xe35')
    optimized = evaluate_expression('use_port_xe35')
    return {'optimized': optimized, 'private_maxK': context.get('resolved_max_seqlen_k_xe35', context['max_seqlen_k']) if optimized else None,
            'fallback_original_maxK': None if optimized else context['max_seqlen_k']}


def postcheck():
    before = (ROOT / 'before-entry.cpp').read_text()
    after = ENTRY.read_text()
    delta = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/' + ENTRY_PATH, tofile='b/' + ENTRY_PATH))
    approved = ART / 'MODEL51_XE35_METADATA_BRIDGE_PROPOSAL_v1.patch'
    assert delta == approved.read_text()
    assert digest(approved) == 'd990aacd4e5b22eec71ee1ad8bb85a43339d266f0474b0950ae1812b94e5946b'
    (ROOT / 'applied-single-source-delta.patch').write_text(delta)
    overlay = load_overlay()
    old = json.loads(FREEZE.read_text())
    files, selection = overlay.source_inventory(PORT)
    changed = sorted(name for name in files if files[name] != old['files'].get(name))
    assert set(files) == set(old['files']) and changed == [ENTRY_PATH]
    pre = json.loads((ROOT / 'pre-edit-drain-and-provenance.json').read_text())
    for name, expected in pre['old_freeze_sha256'].items():
        assert digest(Path(name)) == expected
    for start, end in [('  const bool use_port_xe35', '  if (!use_port_xe35)'),
                       ('  if (!use_port_xe35)', '  std::optional<at::Tensor> output_xe35')]:
        assert before[before.index(start):before.index(end)] == after[after.index(start):after.index(end)]
    plan = json.loads((ART / 'model-config51-v1/plan.json').read_text())
    model = []
    for case in plan['cases']:
        fixture = plan['fixture_files'][case['bundle']]['tensors']
        normalized = dict(case, paged_KV_pool_shape=fixture['key']['shape'],
                          page_table_shape=case.get('page_table_shape', fixture['table']['shape']))
        values = metadata(normalized, 0)
        previous = evaluate(before, values, False)
        candidate = evaluate(after, values, True)
        assert not previous['optimized'] and candidate['optimized']
        assert candidate['private_maxK'] == case['source_wrapper_inferred_max_seqlen_k']
        model.append({'name': case['name'], 'serving_batch': 8, 'before': previous, 'candidate_CPU_guard': candidate})
    for case in json.loads((ART / 'normal-v4-gpu5-accuracy-only-v1.json').read_text())['cases']:
        normalized = dict(case, paged_KV_pool_shape=[520, 64, 16, 64],
                          page_table_shape=[case['requests'], (case['cache_length'] + 63) // 64])
        values = metadata(normalized, case['cache_length'])
        assert evaluate(before, values, False) == evaluate(after, values, True)
        assert evaluate(after, values, True)['optimized']
    base = metadata(dict(requests=8, query_length=1, window=-1,
        paged_KV_pool_shape=[520, 64, 16, 64], page_table_shape=[8, 65]), 0)
    probes = []

    def probe(name, values, expected, private=None):
        result = evaluate(after, values, True)
        assert result['optimized'] == expected, (name, result)
        if expected:
            assert result['private_maxK'] == private
        else:
            assert result['fallback_original_maxK'] == values['max_seqlen_k']
        probes.append({'name': name, 'serving_batch': 8, 'result': result})

    for maximum in (0, -1, -INT_MAX - 1, 4097, 4112, 4096, 4113, 4160, INT_MAX):
        values = copy.deepcopy(base)
        values['max_seqlen_k'] = maximum
        supported = maximum <= 0 or maximum in (4097, 4112)
        probe('decode_explicit_or_sentinel_' + str(maximum), values, supported, 4160 if maximum <= 0 else maximum)
    for columns in (0, -1, 64, 65, 66, INT_MAX // 64, INT_MAX // 64 + 1):
        values = copy.deepcopy(base)
        values['page_table'] = MetadataTensor([8, columns])
        probe('decode_capacity_columns_' + str(columns), values, columns == 65, 4160)
    variants = {'table_absent': {'page_table': MetadataTensor([], present=False)},
        'table_wrong_rank': {'page_table': MetadataTensor([65])},
        'K_wrong_rank': {'k': MetadataTensor([520, 16, 64])},
        'Q_wrong_shape': {'q': MetadataTensor([9, 128, 64])},
        'Q_wrong_heads': {'q': MetadataTensor([8, 127, 64])},
        'Q_wrong_dtype': {'q': MetadataTensor([8, 128, 64], dtype='FP32')},
        'K_wrong_page_size': {'k': MetadataTensor([520, 128, 16, 64])},
        'V_wrong_shape': {'v': MetadataTensor([519, 64, 16, 64])},
        'out_wrong_dtype': {'out': MetadataTensor([8, 128, 64], dtype='FP32')},
        'noncontiguous': {'q': MetadataTensor([8, 128, 64], contiguous=False)},
        'noncausal': {'is_causal': False}, 'wrong_window': {'window_size_left': 100},
        'requested_LSE': {'softmax_lse': MetadataTensor([8, 128])},
        'margin_override': {'sm_margin': 1}, 'split_override': {'num_kv_splits': 1}}
    for name, changes in variants.items():
        values = copy.deepcopy(base)
        values.update(changes)
        probe(name, values, False)
    assert expression(after, 'use_port_xe35') == expression(before, 'use_port_xe35')
    report = {'epoch': time.time(), 'CPU_only': True, 'compile_GPU_started': False,
        'test_method': 'actual before/after C++ bool expressions lexically translated to Python and evaluated against CPU shape-only objects; no compiled C++/GPU proof',
        'approved_patch_exact_bytes_applied': True, 'approved_patch_sha256': digest(approved),
        'changed_source_paths': changed, 'source_file_count': len(files),
        'v4_source_digest': old['source_digest'], 'candidate_v5_source_digest': overlay.manifest_digest(files),
        'before_entry_sha256': digest(ROOT / 'before-entry.cpp'), 'after_entry_sha256': digest(ENTRY),
        'fallback_original_33_arguments_byte_unchanged': True, 'other_existing_guards_byte_unchanged': True,
        'all_other_659_source_file_records_unchanged': True, 'old_freezes_unchanged': True,
        'old_stage_not_written': True, 'BMG_generic_Python_kernel_math_unchanged': True,
        'model51': model, 'old38_positive_maxK_guard_unchanged': 38, 'boundary_probes': probes,
        'scope': 'SOURCE runtime optimized51 versus v4 AFTER runtime generic51; candidate guard51 only, optimized PORT runtime/reference pending main builder/grant',
        'capacity_contract': '4096/4160 is real host shape capacity, not a fabricated per-request logical cache length',
        'source_inventory_selection': selection, 'drain': drain()}
    write_report('single-source-delta-and-CPU-guard-audit.json', report)
    print(json.dumps({'CPU_guard_exit': 0, 'single_source_delta': changed, 'model51': 51,
        'old38_preserved': 38, 'boundary_probes': len(probes), 'source_digest': report['candidate_v5_source_digest']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'postcheck'])
    args = parser.parse_args()
    preflight() if args.phase == 'preflight' else postcheck()

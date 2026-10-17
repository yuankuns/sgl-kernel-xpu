import copy
import difflib
import hashlib
import json
from pathlib import Path
import time


ART = Path(__file__).resolve().parent
PORT = ART.parent.parent / 'pr/muse-fmha-xe35'
SOURCE = ART / 'matched-native-icpx-v1/source'
ENTRY = PORT / 'src/sycl/xe35/fmha/entry.cpp'
INT_MAX = 2147483647


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_guard(case, maximum, proposed=False, capacity_hint=False):
    if case['phase'] == 'decode':
        return case['query_length'] == 1 and case['requests'] == 8 and (
            4097 <= maximum <= 4112 or proposed and capacity_hint and maximum == 4160)
    batches = (1, 3, 5, 7, 8) if proposed else (1, 7, 8)
    return case['query_length'] == 4096 and case['requests'] in batches and maximum == 4096


def resolve_capacity(case, original):
    if original > 0:
        return original, False
    page_shape = case.get('page_table_shape', [])
    cache_shape = case.get('paged_KV_pool_shape', [])
    valid = len(page_shape) == 2 and len(cache_shape) == 4 and cache_shape[1] == 64
    valid = valid and 0 < page_shape[1] <= INT_MAX // 64
    if not valid:
        return original, False
    return page_shape[1] * cache_shape[1], True


def route(case, original, proposed=False):
    maximum, capacity_hint = resolve_capacity(case, original) if proposed else (original, False)
    shape = production_guard(case, maximum, proposed, capacity_hint)
    supported = case.get('other_existing_port_guards', True) and case.get('architecture', '3.5') == '3.5'
    return {'optimized': shape and supported, 'private_maxK': maximum if shape and supported else None,
            'fallback_maxK': original if not (shape and supported) else None,
            'capacity_not_logical_length': capacity_hint}


def proposed_patch(original):
    old_shape = '''  const bool production_shape_xe35 =
      (max_seqlen_q == 1 && batch_xe35 == 8 && max_seqlen_k >= 4097 && max_seqlen_k <= 4112) ||
      (max_seqlen_q == 4096 && (batch_xe35 == 1 || batch_xe35 == 7 || batch_xe35 == 8) && max_seqlen_k == 4096);'''
    new_shape = '''  const bool capacity_hint_xe35 = max_seqlen_k <= 0 && page_table.has_value() &&
      page_table->dim() == 2 && k.dim() == 4 && k.size(1) == 64 &&
      page_table->size(1) > 0 && page_table->size(1) <= std::numeric_limits<int>::max() / 64;
  const int resolved_max_seqlen_k_xe35 = capacity_hint_xe35
      ? static_cast<int>(page_table->size(1) * k.size(1)) : max_seqlen_k;
  const bool production_shape_xe35 =
      (max_seqlen_q == 1 && batch_xe35 == 8 &&
       ((resolved_max_seqlen_k_xe35 >= 4097 && resolved_max_seqlen_k_xe35 <= 4112) ||
        (capacity_hint_xe35 && resolved_max_seqlen_k_xe35 == 4160))) ||
      (max_seqlen_q == 4096 &&
       (batch_xe35 == 1 || batch_xe35 == 3 || batch_xe35 == 5 || batch_xe35 == 7 || batch_xe35 == 8) &&
       resolved_max_seqlen_k_xe35 == 4096);'''
    assert original.count(old_shape) == 1
    candidate = '#include <limits>\n' + original.replace(old_shape, new_shape)
    old_call = '''  mha_fwd_xe35(
      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,'''
    new_call = '''  mha_fwd_xe35(
      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, resolved_max_seqlen_k_xe35,'''
    assert candidate.count(old_call) == 1
    candidate = candidate.replace(old_call, new_call)
    for start, end in [('  const bool use_port_xe35', '  if (!use_port_xe35)'),
                       ('  if (!use_port_xe35)', '  std::optional<at::Tensor> output_xe35')]:
        assert original[original.index(start):original.index(end)] == candidate[candidate.index(start):candidate.index(end)]
    return ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
        fromfile='a/src/sycl/xe35/fmha/entry.cpp', tofile='b/src/sycl/xe35/fmha/entry.cpp'))


def main():
    plan_path = ART / 'model-config51-v1/plan.json'
    representative_path = ART / 'normal-v4-gpu5-accuracy-only-v1.json'
    audit_path = ART / 'MODEL51_SOURCE_DISPATCH_NATIVE_COMPILER_AUDIT_v1.json'
    runtime_path = ART / 'MODEL51_SOURCE_ROUTE_RUNTIME_BINDING_CPU_v1.json'
    source_wrapper = SOURCE / 'site/sgl_kernel/flash_attn.py'
    source_root = SOURCE / 'source/src/sycl/flash_attentionXe35.cpp'
    source_split = SOURCE / 'source/src/sycl/kernels/flash_attention_v2/xe3/kernel/xe_fmha_fwd_decode_splitkv_kernel.hpp'
    paths = [plan_path, representative_path, audit_path, runtime_path, ENTRY, source_wrapper, source_root, source_split]
    before = {str(path): digest(path) for path in paths}
    audit = json.loads(audit_path.read_text())
    binding = json.loads(runtime_path.read_text())
    assert audit['SOURCE_prefill_optimized_cases'] == 6 and audit['SOURCE_decode_optimized_cases'] == 45
    assert binding['runtime_actual_mapped_3_tested_DSO_matches_CPU_route_proof']
    assert 'max_seqlen_k = int(page_table.shape[1] * k_cache.shape[1])' in source_wrapper.read_text()
    assert 'const int seq_len_kv_cache = s.seq_len_kv_cache.cumulative_length[idx_b];' in source_split.read_text()
    plan = json.loads(plan_path.read_text())
    cases = copy.deepcopy(plan['cases'])
    for case in cases:
        tensors = plan['fixture_files'][case['bundle']]['tensors']
        case.setdefault('page_table_shape', tensors['table']['shape'])
        case.setdefault('paged_KV_pool_shape', tensors['key']['shape'])
    assert len(cases) == 51 and all(case['serving_batch'] == 8 for case in cases)
    rows = []
    for case in cases:
        source_maximum, inferred = resolve_capacity(case, 0)
        assert inferred and source_maximum == case['source_wrapper_inferred_max_seqlen_k']
        current = route(case, 0)
        proposed = route(case, 0, True)
        assert not current['optimized'] and current['fallback_maxK'] == 0
        assert proposed['optimized'] and proposed['private_maxK'] == source_maximum
        rows.append({'name': case['name'], 'serving_batch': 8, 'active_sequences': case['requests'],
            'actual_logical_cache_length_from_immutable_CPU_plan': case['cache_length'],
            'actual_SOURCE_native_route': 'optimized ' + case['phase'], 'SOURCE_maxK': source_maximum,
            'frozen_PORT': current, 'proposed_PORT_metadata_contract_not_executed': proposed})
    representative = json.loads(representative_path.read_text())['cases']
    assert len(representative) == 38
    for case in representative:
        normalized = dict(case, phase='decode' if case['query_length'] == 1 else 'prefill')
        assert route(normalized, case['cache_length'])['optimized']
        assert route(normalized, case['cache_length']) == route(normalized, case['cache_length'], True)
    negatives = []
    reference = cases[6]
    assert reference['phase'] == 'decode'
    variants = [('non_Xe35_owner', {'architecture': '2.0'}, 0),
        ('advanced_or_dtype_or_layout_existing_guard_false', {'other_existing_port_guards': False}, 0),
        ('missing_table', {'page_table_shape': []}, 0),
        ('bad_table_rank', {'page_table_shape': [65]}, 0),
        ('empty_columns', {'page_table_shape': [8, 0]}, 0),
        ('int_capacity_overflow', {'page_table_shape': [8, INT_MAX]}, 0),
        ('outside_B8_page_capacity', {'page_table_shape': [8, 66]}, 0),
        ('non_production_query_length', {'query_length': 2}, 0),
        ('positive_capacity_hint_not_newly_enabled', {}, 4160),
        ('positive_outside_production_interval', {}, 4113)]
    for name, changes, original in variants:
        probe = copy.deepcopy(reference)
        probe.update(changes)
        result = route(probe, original, True)
        assert not result['optimized'] and result['fallback_maxK'] == original
        negatives.append({'name': name, 'serving_batch': 8, 'result': result})
    patch = proposed_patch(ENTRY.read_text())
    patch_path = ART / 'MODEL51_XE35_METADATA_BRIDGE_PROPOSAL_v1.patch'
    patch_path.write_text(patch)
    after = {str(path): digest(path) for path in paths}
    assert before == after
    report = {'epoch': time.time(), 'CPU_only': True, 'GPU_or_compile_started': False,
        'patch_applied': False, 'main_review_required': True,
        'gate_kind': 'stdlib metadata-contract simulation plus unchanged source/runtime/native provenance; not compiled C++ or optimized PORT runtime proof',
        'source_head': audit['source_head'], 'source_files_before_sha256': before,
        'source_files_after_sha256': after, 'proposed_patch_sha256': digest(patch_path),
        'model51_cases': rows, 'existing38_guard_and_positive_maxK_unchanged': 38,
        'B8_only_negative_contract_cases': negatives,
        'SOURCE_optimized_prefill': 6, 'SOURCE_optimized_decode': 45, 'frozen_PORT_generic': 51,
        'proposed_PORT_optimized_predicates_not_runtime_validation': 51,
        'capacity_decode_contract_change_explicit': 'sentinel<=0, true table capacity4160; not a host assertion of logical cache length4097..4112',
        'review_and_new_freeze_current_install_route_fullrefs_required': True}
    (ART / 'MODEL51_XE35_METADATA_BRIDGE_CPU_CONTRACT_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'CPU_contract_exit': 0, 'model51': 51, 'old38_preserved': 38,
        'negative_cases': len(negatives), 'patch_applied': False, 'GPU_or_compile_started': False}))


if __name__ == '__main__':
    main()

import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from input_forensics import tensor_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bits32(value):
    return int(value.reshape(1).float().view(torch.int32)[0]) & 4294967295


def main():
    directory = Path(__file__).resolve().parent
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80)) and not torch.xpu.is_initialized()
    runtime_pid = int((directory / 'container-runtime.pid').read_text())
    assert not Path('/proc', str(runtime_pid)).exists()
    evidence_path = directory / 'first-failure-six-production-shapes-full-local-tensors.pt'
    evidence = torch.load(evidence_path, map_location='cpu', weights_only=True, mmap=True)
    prefix = next(name[:-len('logits_cpu')] for name in evidence if name.endswith('-logits_cpu'))
    fields = {name: prefix + name for name in ['logits_cpu', 'bias_cpu', 'logits', 'bias', 'scores', 'ranking', 'ids', 'weights']}
    cache_path = directory / 'production-cpu-cache-v2/route-32768.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    for name, original in [('logits_cpu', 'logits'), ('bias_cpu', 'bias'), ('logits', 'logits'), ('bias', 'bias')]:
        assert tensor_sha256(evidence[fields[name]]) == tensor_sha256(cache[original])
    ids = evidence[fields['ids']]
    weights = evidence[fields['weights']]
    expected_ids = cache['CPU_reference_ids']
    device_scores = evidence[fields['scores']]
    device_rank = evidence[fields['ranking']]
    scaled_cpu = cache['logits'] * 0.3535533905932738
    cpu_scores = torch.sigmoid(scaled_cpu)
    cpu_rank = cpu_scores + cache['bias']
    recomputed_ids = torch.topk(cpu_rank, 8, dim=-1, sorted=True).indices.int()
    assert torch.equal(recomputed_ids, expected_ids)
    cpu_selected = cpu_scores.gather(1, recomputed_ids.long())
    cpu_weights = cpu_selected / (cpu_selected.sum(dim=-1, keepdim=True) + 1e-15)
    assert torch.equal(cpu_weights, cache['CPU_reference_weights'])
    coordinates = (ids != expected_ids).nonzero().tolist()
    rows = sorted(set(row for row, slot in coordinates))
    native_sorted = torch.sort(ids, dim=-1).values
    cpu_sorted = torch.sort(expected_ids, dim=-1).values
    set_different_rows = torch.any(native_sorted != cpu_sorted, dim=-1).nonzero().flatten().tolist()
    device_selected = device_scores.gather(1, ids.long())
    device_expected_weights = device_selected / (device_selected.sum(dim=-1, keepdim=True) + 1e-15)
    original_device_rank_diff = torch.count_nonzero(device_rank.gather(1, ids.long()) != torch.topk(device_rank, 8, dim=-1, sorted=True).values).item()
    original_device_weight_max = float((weights - device_expected_weights).abs().max())
    candidate_rows = []
    details = []
    for row in rows:
        scalar64_scores = [1.0 / (1.0 + math.exp(-float(cache['logits'][row, expert]) * 0.3535533905932738)) for expert in range(256)]
        scalar64_rank = [scalar64_scores[expert] + float(cache['bias'][expert]) for expert in range(256)]
        rounded32_input_scores64 = [1.0 / (1.0 + math.exp(-float(scaled_cpu[row, expert]))) for expert in range(256)]
        rounded32_input_rank64 = [rounded32_input_scores64[expert] + float(cache['bias'][expert]) for expert in range(256)]
        top64 = sorted(range(256), key=lambda expert: (-scalar64_rank[expert], expert))[:8]
        top32_input64 = sorted(range(256), key=lambda expert: (-rounded32_input_rank64[expert], expert))[:8]
        candidates = []
        for expert in range(256):
            candidates.append({'expert': expert, 'raw_logit': float(cache['logits'][row, expert]),
                'raw_logit_bits': bits32(cache['logits'][row, expert]), 'raw_bias': float(cache['bias'][expert]),
                'raw_bias_bits': bits32(cache['bias'][expert]), 'FP32_scaled_logit': float(scaled_cpu[row, expert]),
                'CPU_FP32_sigmoid': float(cpu_scores[row, expert]), 'CPU_FP32_sigmoid_bits': bits32(cpu_scores[row, expert]),
                'saved_PyTorch_XPU_sigmoid': float(device_scores[row, expert]), 'saved_PyTorch_XPU_sigmoid_bits': bits32(device_scores[row, expert]),
                'CPU_FP32_ranking': float(cpu_rank[row, expert]), 'CPU_FP32_ranking_bits': bits32(cpu_rank[row, expert]),
                'saved_PyTorch_XPU_ranking': float(device_rank[row, expert]), 'saved_PyTorch_XPU_ranking_bits': bits32(device_rank[row, expert]),
                'independent_scalar_FP64_sigmoid': scalar64_scores[expert], 'independent_scalar_FP64_ranking': scalar64_rank[expert],
                'FP64_on_production_FP32_scaled_input_ranking': rounded32_input_rank64[expert]})
        candidate_rows.append({'row': row, 'all256_candidates': candidates})
        native_set = set(ids[row].tolist())
        cpu_set = set(expected_ids[row].tolist())
        pairs = []
        for current_row, slot in coordinates:
            if current_row != row:
                continue
            native_id = int(ids[row, slot])
            cpu_id = int(expected_ids[row, slot])
            pairs.append({'slot': slot, 'actual_native_id': native_id, 'CPU_cache_id': cpu_id,
                'CPU_FP32_gap_CPU_id_minus_native_id': float(cpu_rank[row, cpu_id] - cpu_rank[row, native_id]),
                'CPU_FP32_ranking_bit_distance': abs(bits32(cpu_rank[row, cpu_id]) - bits32(cpu_rank[row, native_id])),
                'CPU_FP32_ranking_tie': bool(cpu_rank[row, cpu_id] == cpu_rank[row, native_id]),
                'saved_PyTorch_XPU_gap_CPU_id_minus_native_id': float(device_rank[row, cpu_id] - device_rank[row, native_id]),
                'saved_PyTorch_XPU_ranking_bit_distance': abs(bits32(device_rank[row, cpu_id]) - bits32(device_rank[row, native_id])),
                'saved_PyTorch_XPU_ranking_tie': bool(device_rank[row, cpu_id] == device_rank[row, native_id]),
                'independent_FP64_gap_CPU_id_minus_native_id': scalar64_rank[cpu_id] - scalar64_rank[native_id],
                'FP64_on_FP32_scaled_input_gap_CPU_id_minus_native_id': rounded32_input_rank64[cpu_id] - rounded32_input_rank64[native_id],
                'actual_native_weight': float(weights[row, slot]), 'CPU_cache_weight': float(cache['CPU_reference_weights'][row, slot]),
                'ordered_weight_error': abs(float(weights[row, slot] - cache['CPU_reference_weights'][row, slot]))})
        details.append({'row': row, 'actual_native_ordered_ids': ids[row].tolist(), 'CPU_cache_ordered_ids': expected_ids[row].tolist(),
            'independent_scalar_FP64_ordered_ids': top64, 'FP64_on_FP32_scaled_input_ordered_ids': top32_input64,
            'native_only_selected': sorted(native_set - cpu_set), 'CPU_only_selected': sorted(cpu_set - native_set), 'cutoff_pairs': pairs})
    (directory / 'CPU-first-failure-all256-candidates.json').write_text(json.dumps(candidate_rows, indent=2) + '\n')
    torch.save({'actual_native_ids': ids, 'actual_native_weights': weights, 'saved_PyTorch_XPU_sigmoid': device_scores,
                'saved_PyTorch_XPU_ranking': device_rank, 'CPU_FP32_sigmoid': cpu_scores, 'CPU_FP32_ranking': cpu_rank,
                'CPU_FP32_ids': recomputed_ids, 'CPU_FP32_weights': cpu_weights, 'original_raw_logits': cache['logits'],
                'original_raw_bias': cache['bias']}, directory / 'CPU-first-failure-full-reference-comparison.pt')
    report = {'epoch': time.time(), 'CPU_classification_actual_exit': 0, 'GPU_original_failure_actual_exit_preserved': 1,
        'container_runtime_PID_absent': runtime_pid, 'saved_tensor_fieldnames': fields,
        'original_device_reference_contract': 'exact equality of selected biased ranking VALUES to topk VALUES on saved PyTorch XPU reference; unbiased weights for actual native IDs atol2.5e-4 rtol0; not arbitrary CPU exact-ID identity',
        'new_fullCPU_exact_ID_contract': 'CPU FP32 multiply -> torch.sigmoid CPU -> FP32 bias-add -> torch.topk(sorted=True); native ordered IDs exact equal to CPU topk IDs, weights from CPU-selected IDs',
        'independent_FP64_contract': 'Python math.exp binary64 on raw logit times binary64 multiplier, bias-add binary64; deterministic expert-index secondary ordering only for diagnostic, not substituted acceptance',
        'ordered_ID_different_elements': len(coordinates), 'ordered_ID_total_elements': ids.numel(),
        'all_selected_set_different_rows': set_different_rows, 'all_selected_set_total_rows': ids.shape[0],
        'original_device_reference_exact_rank_value_different_elements': original_device_rank_diff,
        'original_device_reference_actual_ID_weight_max_abs_error': original_device_weight_max,
        'new_CPU_ordered_weight_max_abs_error': float((weights - cache['CPU_reference_weights']).abs().max()),
        'details': details, 'original14_tests_executed': 0, 'original1300_tests_executed': 0,
        'new_fullCPU_cache_and_actualGPU_input_identity_rawSHA': {'logits': tensor_sha256(cache['logits']), 'bias': tensor_sha256(cache['bias'])},
        'cache_file_SHA256': digest(cache_path), 'failure_full_tensor_file_SHA256': digest(evidence_path),
        'all256_candidates_SHA256': digest(directory / 'CPU-first-failure-all256-candidates.json'),
        'full_CPU_reference_comparison_SHA256': digest(directory / 'CPU-first-failure-full-reference-comparison.pt'),
        'native_internal_sigmoid_ranking_NOT_observed': True,
        'limitation': 'Saved XPU sigmoid/ranking is real PyTorch-device reference, not native-internal intermediate. FP32/FP64 rank differences alone do not prove native implementation bug or exempt new fullCPU gate.',
        'GPU_initialized': False, 'no_GPU_retry_compile_math_tolerance_or_PR_changes': True}
    assert not torch.xpu.is_initialized()
    (directory / 'CPU-first-failure-classification-v2.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ['new_fullCPU_cache_and_actualGPU_input_identity_rawSHA']}, indent=2))


if __name__ == '__main__':
    main()

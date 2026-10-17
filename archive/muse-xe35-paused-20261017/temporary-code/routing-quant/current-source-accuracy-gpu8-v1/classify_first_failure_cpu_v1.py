import hashlib
import json
import os
from pathlib import Path

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bits(value):
    return int(value.reshape(1).float().view(torch.int32)[0]) & 4294967295


def main():
    directory = Path(__file__).resolve().parent
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80)) and not torch.xpu.is_initialized()
    evidence_path = directory / 'first-failure-six-production-shapes-full-local-tensors.pt'
    evidence = torch.load(evidence_path, map_location='cpu', weights_only=True, mmap=True)
    cache_path = directory / 'production-cpu-cache-v2/route-32768.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    for name in ['logits_cpu', 'bias_cpu']:
        assert torch.equal(evidence['frame0-' + name], cache['logits' if name == 'logits_cpu' else 'bias'])
    ids = evidence['frame0-ids']
    expected = cache['CPU_reference_ids']
    coordinates = (ids != expected).nonzero().tolist()
    cpu_scores = torch.sigmoid(cache['logits'] * 0.3535533905932738)
    cpu_rank = cpu_scores + cache['bias']
    device_scores = evidence['frame0-scores']
    device_rank = evidence['frame0-ranking']
    points = []
    for row, slot in coordinates:
        actual_id = int(ids[row, slot])
        expected_id = int(expected[row, slot])
        experts = sorted(set([actual_id, expected_id]))
        values = []
        for expert in experts:
            values.append({'expert': expert, 'input_logit': float(cache['logits'][row, expert]),
                           'bias': float(cache['bias'][expert]),
                           'CPU_sigmoid': float(cpu_scores[row, expert]), 'CPU_sigmoid_bits': bits(cpu_scores[row, expert]),
                           'saved_device_torch_sigmoid': float(device_scores[row, expert]),
                           'saved_device_torch_sigmoid_bits': bits(device_scores[row, expert]),
                           'CPU_ranking': float(cpu_rank[row, expert]), 'CPU_ranking_bits': bits(cpu_rank[row, expert]),
                           'saved_device_torch_ranking': float(device_rank[row, expert]),
                           'saved_device_torch_ranking_bits': bits(device_rank[row, expert])})
        points.append({'row': row, 'slot': slot, 'actual_native_id': actual_id, 'fullCPU_reference_id': expected_id,
                       'experts': values,
                       'CPU_ranking_tie': bool(cpu_rank[row, actual_id] == cpu_rank[row, expected_id]),
                       'saved_device_torch_ranking_tie': bool(device_rank[row, actual_id] == device_rank[row, expected_id])})
    device_top_values = torch.topk(device_rank, 8, dim=-1, sorted=True).values
    selected_device_values = device_rank.gather(1, ids.long())
    original_ranking_gate_different_elements = torch.count_nonzero(selected_device_values != device_top_values).item()
    weights = evidence['frame0-weights']
    selected = device_scores.gather(1, ids.long())
    device_expected_weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-15)
    original_weight_gate_max_error = float((weights - device_expected_weights).abs().max())
    report = {'CPU_classification_actual_exit': 0, 'actual_GPU_gate_exit_preserved': 1,
              'failed_gate': 'new independent exact fullCPU ID equality, not original pytest or original device-ranking gate',
              'mismatch_count': len(coordinates), 'total_ids': ids.numel(), 'points': points,
              'saved_device_torch_reference_original_exact_rank_value_gate_different_elements': original_ranking_gate_different_elements,
              'saved_device_torch_reference_original_weight_atol': 2.5e-4,
              'saved_device_torch_reference_original_weight_max_error': original_weight_gate_max_error,
              'full_failed_input_and_native_output_saved': True, 'evidence_file_SHA256': digest(evidence_path),
              'new_full_CPU_cache_SHA256': digest(cache_path), 'GPU_initialized': False,
              'original14_and1300_tests_executed': False, 'all6_shapes_passed': False,
              'unproved': 'saved device scores/ranking are the original PyTorch XPU reference, not directly observed internal native ranking intermediates; no native-internal exp/rounding attribution claimed',
              'no_new_GPU_retry_math_tolerance_or_PR_change': True}
    assert not torch.xpu.is_initialized()
    (directory / 'CPU-first-failure-classification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

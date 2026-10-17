import hashlib
import json
import os
from pathlib import Path
import time

import torch
from input_forensics import tensor_sha256
from routing_contract import verify_routing


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    previous = directory.parent / 'current-source-accuracy-gpu8-v1'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    evidence_path = previous / 'CPU-first-failure-full-reference-comparison.pt'
    evidence = torch.load(evidence_path, map_location='cpu', weights_only=True, mmap=True)
    actual = verify_routing(evidence['original_raw_logits'], evidence['original_raw_bias'], evidence['actual_native_ids'], evidence['actual_native_weights'])
    cpu_rank = evidence['CPU_FP32_ranking']
    device_rank = evidence['saved_PyTorch_XPU_ranking']
    actual_ids = evidence['actual_native_ids'].long()
    selected_cpu = cpu_rank.gather(1, actual_ids)
    selected_device = device_rank.gather(1, actual_ids)
    selected_differences = selected_cpu.contiguous().view(torch.int32) != selected_device.contiguous().view(torch.int32)
    candidate_differences = cpu_rank.contiguous().view(torch.int32) != device_rank.contiguous().view(torch.int32)
    logits = torch.zeros(1, 16, dtype=torch.float32)
    bias = torch.tensor([[8., 7., 6., 5., 4., 3., 2., 1., 1., 0., -1., -2., -3., -4., -5., -6.]], dtype=torch.float32).flatten()
    rank = torch.sigmoid(logits * 0.3535533905932738) + bias
    ids = rank.topk(8, dim=1, sorted=True).indices.int()
    replacement = 8 if int(ids[0, -1]) == 7 else 7
    tie_ids = ids.clone()
    tie_ids[0, -1] = replacement
    scores = torch.sigmoid(logits * 0.3535533905932738)
    weights = scores.gather(1, ids.long())
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-15)
    verify_routing(logits, bias, ids, weights)
    verify_routing(logits, bias, tie_ids, weights)
    negative_cases = {}
    wrong_ids = ids.clone()
    wrong_ids[0, -1] = 9
    duplicate = ids.clone()
    duplicate[0, -1] = duplicate[0, 0]
    outside = ids.clone()
    outside[0, -1] = 16
    near_bias = bias.clone()
    near_bias[replacement] = torch.nextafter(near_bias[replacement], torch.tensor(float('-inf')))
    near_bias[replacement] = torch.nextafter(near_bias[replacement], torch.tensor(float('-inf')))
    for name, candidate_ids, candidate_weights, candidate_bias in [
        ('non_tie', wrong_ids, weights, bias), ('duplicate', duplicate, weights, bias),
        ('out_of_range', outside, weights, bias), ('wrong_weights', ids, weights + .001, bias),
        ('close_but_non_bit_equal', tie_ids, weights, near_bias)]:
        try:
            verify_routing(logits, candidate_bias, candidate_ids, candidate_weights)
        except AssertionError as error:
            negative_cases[name] = str(error)
        else:
            raise AssertionError(f'negative contract incorrectly accepted: {name}')
    assert int((previous / 'host-launch.exit').read_text()) == 1
    report = {'epoch': time.time(), 'actual_CPU_contract_exit': 0, 'v1_GPU_failure_exit_preserved': 1,
              'whole_saved_32768_rows_corrected_contract': actual,
              'all_candidate_CPU_vs_saved_PyTorch_XPU_rank_bit_differences': int(candidate_differences.sum()),
              'all_candidate_CPU_vs_saved_PyTorch_XPU_rank_max_abs_error': float((cpu_rank - device_rank).abs().max()),
              'actual_selected_CPU_vs_saved_PyTorch_XPU_rank_bit_differences': int(selected_differences.sum()),
              'negative_cases_rejected': negative_cases, 'bit_equal_cutoff_swap_passed': True,
              'original_input_cache_identity_SHA': {name: tensor_sha256(evidence[field]) for name, field in [('logits', 'original_raw_logits'), ('bias', 'original_raw_bias')]},
              'saved_full_evidence_SHA256': digest(evidence_path),
              'native_internal_scores_not_observed': True, 'GPU_initialized': False,
              'no_native_PR_math_precision_changes': True}
    (directory / 'CPU-contract-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert int(selected_differences.sum()) == 0, 'report CPU-vs-GPU selected value differences to main before GPU grant'
    assert not torch.xpu.is_initialized()


if __name__ == '__main__':
    main()

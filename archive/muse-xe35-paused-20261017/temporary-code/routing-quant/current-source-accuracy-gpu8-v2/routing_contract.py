import torch


def verify_routing(logits, bias, ids, weights):
    assert logits.dtype == torch.float32 and bias.dtype == torch.float32
    assert ids.dtype in (torch.int32, torch.int64)
    assert ids.shape == weights.shape == (logits.shape[0], 8)
    assert torch.isfinite(logits).all() and torch.isfinite(bias).all()
    assert torch.isfinite(weights).all()
    assert ((ids >= 0) & (ids < logits.shape[1])).all(), 'out-of-range expert'
    ordered_set = ids.sort(dim=1).values
    assert (ordered_set[:, 1:] != ordered_set[:, :-1]).all(), 'duplicate expert'
    scores = torch.sigmoid(logits * 0.3535533905932738)
    ranking = scores + bias
    top = torch.topk(ranking, 8, dim=-1, sorted=True)
    actual_values = ranking.gather(1, ids.long())
    assert torch.equal(actual_values.contiguous().view(torch.int32), top.values.contiguous().view(torch.int32)), 'non-bit-equal selected ranking'
    changed = ids.long() != top.indices
    assert torch.equal(actual_values[changed].contiguous().view(torch.int32), ranking.gather(1, top.indices)[changed].contiguous().view(torch.int32)), 'non-tie ID substitution'
    selected_scores = scores.gather(1, ids.long())
    expected_weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-15)
    torch.testing.assert_close(weights, expected_weights, atol=2.5e-4, rtol=0)
    return {'rows': logits.shape[0], 'candidates': logits.numel(), 'selected_elements': ids.numel(),
            'ordered_ID_differences_allowed_only_at_bit_equal_rank': int(changed.sum()),
            'selected_set_different_rows': int((ids.sort(dim=1).values.long() != top.indices.sort(dim=1).values).any(dim=1).sum()),
            'selected_rank_bit_differences': 0, 'weight_max_abs_error': float((weights - expected_weights).abs().max()),
            'weight_atol': 2.5e-4, 'weight_rtol': 0,
            'weight_reference': 'CPU unbiased sigmoid gather(actual native IDs), row normalization; no arbitrary tie-break weights'}

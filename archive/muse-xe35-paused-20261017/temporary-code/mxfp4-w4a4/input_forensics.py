import hashlib

import torch


def tensor_sha256(tensor):
    flat = tensor.reshape(-1)
    chunk_elements = max(1, (64 * 1024 * 1024) // tensor.element_size())
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), chunk_elements):
        chunk = flat[offset:offset + chunk_elements].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return {'sha256': digest.hexdigest(), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'bytes': tensor.numel() * tensor.element_size()}


def fp64_difference_points(output, activations, activation_scales, weights, weight_scales, counts, examples, source_sample, port_sample):
    points = {}
    selected = []
    row_start = 0
    scale_start = 0
    for expert, count in enumerate(counts):
        padded = (count + 3) // 4 * 4
        if count:
            local_rows = torch.linspace(0, count - 1, min(count, 4)).long().tolist()
            selected.extend((expert, row_start + row, row, scale_start, padded) for row in local_rows)
        row_start += count
        scale_start += padded
    lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    groups = activations.shape[1] * 2 // 32
    for example in examples:
        sample_row, column = example['index']
        expert, row, local_row, scale_offset, padded = selected[sample_row]
        packed_a = activations[row].cpu()
        packed_b = weights[expert, column].cpu()
        codes_a = torch.stack((packed_a & 15, packed_a >> 4), -1).flatten()
        codes_b = torch.stack((packed_b & 15, packed_b >> 4), -1).flatten()
        scales_a = activation_scales.flatten()[scale_offset * groups:(scale_offset + padded) * groups].view(groups, padded)[:, local_row].cpu()
        scales_b = weight_scales[expert, :, column].cpu()
        values_a = lookup[codes_a.long()] * torch.exp2(scales_a.double() - 127).repeat_interleave(32)
        values_b = lookup[codes_b.long()] * torch.exp2(scales_b.double() - 127).repeat_interleave(32)
        expected = float(torch.sum(values_a * values_b))
        old_source = float(source_sample[sample_row, column])
        old_port = float(port_sample[sample_row, column])
        midpoint = (old_source + old_port) / 2
        points[f'{sample_row}-{column}'] = {'expert': expert, 'global_row': row, 'column': column, 'expected_cpu_fp64': expected, 'current_output': float(output[row, column].cpu()), 'old_source': old_source, 'old_port': old_port, 'old_bf16_midpoint': midpoint, 'expected_minus_midpoint': expected - midpoint, 'expected_bf16_round_to_nearest_even': float(torch.tensor(expected, dtype=torch.float64).bfloat16())}
    return points

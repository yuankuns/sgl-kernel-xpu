import torch


def mxfp4_small_permutation_xe35(
    topk_ids: torch.Tensor,
    padded_counts: torch.Tensor,
    perm_src: torch.Tensor,
    expanded_to_permuted: torch.Tensor,
    scale_map: torch.Tensor,
    alignment: int,
) -> None:
    torch.ops.sgl_kernel.mxfp4_small_permutation_xe35.default(
        topk_ids, padded_counts, perm_src, expanded_to_permuted, scale_map, alignment
    )

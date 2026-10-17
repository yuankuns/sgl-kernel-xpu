#include "sgl_kernel/xe35/mxfp4_permutation_ops.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void mxfp4_small_permutation(
    torch::Tensor topk_ids,
    torch::Tensor padded_counts,
    torch::Tensor perm_src,
    torch::Tensor expanded_to_permuted,
    torch::Tensor scale_map,
    int64_t alignment) {
  mxfp4_small_permutation_xe35(topk_ids, padded_counts, perm_src, expanded_to_permuted, scale_map, alignment);
}
}

#include <torch/all.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void compute_position_xe35(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens);
SGL_KERNEL_EXPORT void compute_position_with_hint_xe35(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint);

SGL_KERNEL_EXPORT void muse_compute_position(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens) {
  compute_position_xe35(positions, extend_start_loc, prefix_lens, seq_lens);
}

SGL_KERNEL_EXPORT void muse_compute_position_with_hint(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint) {
  compute_position_with_hint_xe35(positions, extend_start_loc, prefix_lens, seq_lens, max_seq_len_hint);
}

}

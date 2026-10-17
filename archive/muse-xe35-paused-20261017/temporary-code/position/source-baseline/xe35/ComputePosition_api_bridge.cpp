#include <torch/all.h>

namespace at::native::xpu {
void muse_compute_position(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens);
void muse_compute_position_with_hint(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint);

void compute_position_xe35(
    torch::Tensor positions, torch::Tensor extend_start_loc, torch::Tensor prefix_lens, torch::Tensor seq_lens) {
  muse_compute_position(positions, extend_start_loc, prefix_lens, seq_lens);
}

void compute_position_with_hint_xe35(
    torch::Tensor positions,
    torch::Tensor extend_start_loc,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    int64_t max_seq_len_hint) {
  muse_compute_position_with_hint(positions, extend_start_loc, prefix_lens, seq_lens, max_seq_len_hint);
}

}

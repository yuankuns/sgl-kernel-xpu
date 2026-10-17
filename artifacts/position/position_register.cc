#include <torch/library.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace at::native::xpu {
void muse_compute_position(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void muse_compute_position_with_hint(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("muse_compute_position(Tensor! positions, Tensor! extend_start_loc, Tensor prefix_lens, Tensor seq_lens) -> ()");
  m.impl("muse_compute_position", torch::kXPU, &at::native::xpu::muse_compute_position);
  m.def("muse_compute_position_with_hint(Tensor! positions, Tensor! extend_start_loc, Tensor prefix_lens, Tensor seq_lens, int max_seq_len_hint) -> ()");
  m.impl("muse_compute_position_with_hint", torch::kXPU, &at::native::xpu::muse_compute_position_with_hint);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

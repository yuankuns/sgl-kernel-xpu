#include <torch/extension.h>

namespace at::native::xpu {
void fused_post_norm_residual_pre_norm(
    torch::Tensor branch, torch::Tensor stream, torch::Tensor post_weight,
    torch::Tensor pre_weight, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor output, torch::Tensor new_stream, double post_eps,
    double pre_eps, bool post_norm_input_scale, bool round_pre_norm_input,
    const std::optional<torch::Tensor>& output_fp32);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, module) {
  module.def(
      "fused_post_norm_residual_pre_norm(Tensor branch, Tensor stream, Tensor post_weight, Tensor pre_weight, "
      "Tensor alpha, Tensor beta, Tensor(a!) output, Tensor(b!) new_stream, float post_eps, float pre_eps, "
      "bool post_norm_input_scale=False, bool round_pre_norm_input=False, Tensor(c!)? output_fp32=None) -> ()");
  module.impl("fused_post_norm_residual_pre_norm", torch::kXPU,
              &at::native::xpu::fused_post_norm_residual_pre_norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}

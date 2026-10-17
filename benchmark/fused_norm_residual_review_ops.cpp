#include <torch/all.h>

#include <optional>

namespace at::native::xpu {
void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_post_norm_residual_pre_norm(
    torch::Tensor branch,
    torch::Tensor stream,
    torch::Tensor post_weight,
    torch::Tensor pre_weight,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor new_stream,
    double post_eps,
    double pre_eps,
    bool post_norm_input_scale,
    bool round_pre_norm_input,
    const std::optional<torch::Tensor>& output_fp32);
}

// Small registration target for targeted source builds. It avoids linking the
// unrelated common_ops dependencies when validating this kernel alone.
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps) -> ()");
  m.impl("rmsnorm", torch::kXPU, &at::native::xpu::rmsnorm);
  m.def(
      "fused_post_norm_residual_pre_norm(Tensor branch, Tensor stream, Tensor post_weight, Tensor pre_weight, Tensor "
      "alpha, Tensor beta, Tensor! output, Tensor! new_stream, float post_eps, float pre_eps, bool "
      "post_norm_input_scale, bool round_pre_norm_input, Tensor(a!)? output_fp32=None) -> ()");
  m.impl("fused_post_norm_residual_pre_norm", torch::kXPU, &at::native::xpu::fused_post_norm_residual_pre_norm);
}

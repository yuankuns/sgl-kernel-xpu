#include "sgl_kernel/xe35/residual_norm_ops.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void fused_post_norm_residual_pre_norm(
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
    const std::optional<torch::Tensor>& output_fp32) {
  fused_post_norm_residual_pre_norm_xe35(
      branch, stream, post_weight, pre_weight, alpha, beta, output, new_stream,
      post_eps, pre_eps, post_norm_input_scale, round_pre_norm_input, output_fp32);
}
}

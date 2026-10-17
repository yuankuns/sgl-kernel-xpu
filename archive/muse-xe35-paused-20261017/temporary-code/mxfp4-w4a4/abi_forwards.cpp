#include "sgl_kernel_ops.h"
#include "sgl_kernel/xe35/mxfp4_w4a4.h"

void moe_grouped_mm_nt_xe35_mxfp4_w4a4(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& rows_per_expert,
    const int64_t output_width,
    const int64_t input_width,
    const int64_t num_experts) {
  moe_grouped_mm_nt_mxfp4_w4a4_xe35(
      output, activations, activation_scales, weights, weight_scales,
      bias, rows_per_expert, output_width, input_width, num_experts);
}

void mxfp4_moe_scale_to_mn_major(
    torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& rows_per_expert) {
  mxfp4_moe_scale_to_mn_major_xe35(output, input, rows_per_expert);
}

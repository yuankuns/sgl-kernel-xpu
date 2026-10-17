#include <sgl_kernel/xe35/nvfp4_woq.h>

SGL_KERNEL_EXPORT void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor& output, const torch::Tensor& activations, const torch::Tensor& weights,
    const torch::Tensor& scales, const torch::Tensor& multipliers,
    const torch::Tensor& problems, const torch::Tensor& offsets, const torch::Tensor& workspace) {
  nvfp4_woq_bf16_grouped_mm_xe35(output, activations, weights, scales, multipliers, problems, offsets, workspace);
}

SGL_KERNEL_EXPORT void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor& output, const torch::Tensor& activations, const torch::Tensor& weights,
    const torch::Tensor& activation_scales, const torch::Tensor& scales,
    const torch::Tensor& multipliers, const torch::Tensor& problems,
    const torch::Tensor& offsets, const torch::Tensor& workspace) {
  nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(output, activations, weights, activation_scales, scales, multipliers, problems, offsets, workspace);
}

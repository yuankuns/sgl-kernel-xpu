#include "sgl_kernel/xe35/bf16_gemm_ops.h"

SGL_KERNEL_EXPORT void bf16_gemm_nt_xpu(
    torch::Tensor& output, const torch::Tensor& mat_a, const torch::Tensor& mat_b, int64_t config) {
  bf16_gemm_nt_xpu_xe35(output, mat_a, mat_b, config);
}

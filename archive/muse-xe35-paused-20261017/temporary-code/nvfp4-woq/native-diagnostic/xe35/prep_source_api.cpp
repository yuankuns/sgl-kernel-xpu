#include <sgl_kernel/xe35/nvfp4_woq.h>

namespace at::native::xpu {
SGL_KERNEL_EXPORT void nvfp4_woq_prep(torch::Tensor& multipliers, torch::Tensor& exponents,
                    const torch::Tensor& scales, const torch::Tensor& globals) {
  nvfp4_woq_prep_xe35(multipliers, exponents, scales, globals);
}
}

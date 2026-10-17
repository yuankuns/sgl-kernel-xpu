#include <torch/extension.h>

namespace at::native::xpu {
void nvfp4_woq_prep(torch::Tensor&, torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
}
void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

PYBIND11_MODULE(source_nvfp4_woq_xe35, module) {
  module.def("prep_xe35", [](torch::Tensor out_c, torch::Tensor out_sfmx,
                            const torch::Tensor& scales, const torch::Tensor& global_scales) {
    at::native::xpu::nvfp4_woq_prep(out_c, out_sfmx, scales, global_scales);
  });
  module.def("bf16_xe35", [](torch::Tensor output, const torch::Tensor& activations,
                            const torch::Tensor& weights, const torch::Tensor& scales,
                            const torch::Tensor& multipliers, const torch::Tensor& problems,
                            const torch::Tensor& offsets, const torch::Tensor& workspace) {
    nvfp4_woq_bf16_grouped_mm(output, activations, weights, scales, multipliers, problems, offsets, workspace);
  });
  module.def("mxfp8_xe35", [](torch::Tensor output, const torch::Tensor& activations,
                             const torch::Tensor& weights, const torch::Tensor& activation_scales,
                             const torch::Tensor& scales, const torch::Tensor& multipliers,
                             const torch::Tensor& problems, const torch::Tensor& offsets,
                             const torch::Tensor& workspace) {
    nvfp4_woq_mxfp8_scaled_grouped_mm(
        output, activations, weights, activation_scales, scales, multipliers, problems, offsets, workspace);
  });
}

#include "sgl_kernel_ops.h"
#include "sgl_kernel/xe35/routing_quant.h"

namespace at::native::xpu {
void topk_sigmoid(
    at::Tensor& weights,
    at::Tensor& ids,
    at::Tensor& logits,
    bool renormalize,
    const c10::optional<at::Tensor>& bias,
    double scaling_factor,
    int64_t shared_experts) {
  TORCH_CHECK(scaling_factor == 1.0 && shared_experts == 0, "diagnostic source ABI supports only original topk defaults");
  topk_sigmoid(weights, ids, logits, renormalize, bias);
}

void muse_spark_moe_routing(
    at::Tensor& weights,
    at::Tensor& ids,
    at::Tensor& logits,
    const c10::optional<at::Tensor>& bias,
    double multiplier,
    double scale,
    double epsilon,
    const c10::optional<at::Tensor>& expert_mapping,
    const c10::optional<at::Tensor>& valid_tokens) {
  muse_spark_moe_routing_xe35(weights, ids, logits, bias, multiplier, scale, epsilon, expert_mapping, valid_tokens);
}

void sgl_per_token_group_quant_fp4(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_scales,
    int64_t group_size,
    double epsilon,
    bool fused_silu,
    const std::optional<at::Tensor>& scale_map) {
  sgl_per_token_group_quant_fp4_xe35(input, output_q, output_scales, group_size, epsilon, fused_silu, scale_map);
}

void sgl_padded_gather_per_token_group_quant_fp4(
    at::Tensor input,
    at::Tensor row_map,
    at::Tensor output_q,
    at::Tensor output_scales,
    int64_t group_size,
    double epsilon,
    const std::optional<at::Tensor>& scale_map) {
  sgl_padded_gather_per_token_group_quant_fp4_xe35(input, row_map, output_q, output_scales, group_size, epsilon, scale_map);
}
}

void mxfp4_moe_scale_to_mn_major(
    torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& rows_per_expert) {
  mxfp4_moe_scale_to_mn_major_xe35(output, input, rows_per_expert);
}

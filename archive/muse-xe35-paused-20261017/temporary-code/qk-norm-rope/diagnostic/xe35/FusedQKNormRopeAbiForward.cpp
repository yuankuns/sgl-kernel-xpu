#include "sgl_kernel/xe35/fused_qk_norm_rope.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

SGL_KERNEL_EXPORT void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim) {
  fused_qk_norm_rope_xe35(
      qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight,
      base, is_neox, position_ids, factor, low, high, attention_factor, rotary_dim);
}

}

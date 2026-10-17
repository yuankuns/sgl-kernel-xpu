#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <optional>
#include <tuple>
#include "sgl_kernel_export.h"

namespace cutlass { struct bfloat16_t; }

SGL_KERNEL_EXPORT void dispatch_xe35_decode_64_bf16_xe35(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float q_descale,
    float k_descale,
    float v_descale,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits);
SGL_KERNEL_EXPORT void dispatch_xe35_prefill_64_bf16_xe35(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float q_descale,
    float k_descale,
    float v_descale,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits);

SGL_KERNEL_EXPORT std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<const at::Tensor>& q_v_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor>& page_table,
    std::optional<const at::Tensor>& kv_batch_idx_,
    std::optional<const at::Tensor>& leftpad_k_,
    std::optional<const at::Tensor>& rotary_cos_,
    std::optional<const at::Tensor>& rotary_sin_,
    std::optional<const at::Tensor>& seqlens_rotary_,
    std::optional<at::Tensor>& q_descale_,
    std::optional<at::Tensor>& k_descale_,
    std::optional<at::Tensor>& v_descale_,
    const float softmax_scale_,
    std::optional<const at::Tensor>& sinks_,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const is_rotary_interleaved,
    std::optional<at::Tensor>& scheduler_metadata_,
    int num_kv_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin,
    std::optional<at::Tensor>& out_) {
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 128 && q.size(2) == 64 && q.scalar_type() == at::kBFloat16,
              "diagnostic production BF16 Hq128 D64 only");
  TORCH_CHECK(page_table.has_value() && k.dim() == 4 && k.size(1) == 64 && k.size(2) == 16,
              "diagnostic production paged Hkv16 only");
  const int diagnostic_batch = static_cast<int>(cu_seqlens_q.size(0) - 1);
  TORCH_CHECK((max_seqlen_q == 1 && diagnostic_batch == 8 && max_seqlen_k >= 4097 && max_seqlen_k <= 4112) ||
              (max_seqlen_q == 4096 && (diagnostic_batch == 1 || diagnostic_batch == 7 || diagnostic_batch == 8) &&
               max_seqlen_k == 4096), "diagnostic B8 production decode or prefill waves only");
  TORCH_CHECK(!q_v_.has_value() && !kv_batch_idx_.has_value() && !leftpad_k_.has_value() &&
              !rotary_cos_.has_value() && !rotary_sin_.has_value() && !sinks_.has_value() &&
              !scheduler_metadata_.has_value() && num_kv_splits == 0 && sm_margin == 0 &&
              is_causal && (window_size_left == -1 || window_size_left == 2047) && window_size_right == 0,
              "unsupported diagnostic option");

  const int batch = static_cast<int>(cu_seqlens_q.size(0) - 1);
  TORCH_CHECK(batch > 0, "flash_attentionXe35: empty batch is not supported");
  const bool is_paged = page_table.has_value();

  TORCH_CHECK(q.dim() == 3, "flash_attentionXe35: q must be (total_q, h, d)");
  TORCH_CHECK(cu_seqlens_q.dim() == 1, "flash_attentionXe35: cu_seqlens_q must be 1D");
  TORCH_CHECK(cu_seqlens_k.dim() == 1, "flash_attentionXe35: cache_seqlens must be 1D");
  TORCH_CHECK(!seqlens_rotary_.has_value(), "flash_attentionXe35: seqlens_rotary is not supported");
  TORCH_CHECK(softcap == 0.0f, "flash_attentionXe35: softcap is not supported");
  const bool has_rotary = rotary_cos_.has_value() || rotary_sin_.has_value();
  TORCH_CHECK(!(has_rotary && is_rotary_interleaved), "flash_attentionXe35: rotary interleaving is not supported");
  auto q_dtype = q.scalar_type();
  const bool is_fp8 = q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e5m2;
  TORCH_CHECK(
      q_dtype == at::ScalarType::BFloat16 || is_fp8,
      "flash_attentionXe35: only bf16, float8_e4m3fn, and float8_e5m2 are supported");
  TORCH_CHECK(!(is_fp8 && sinks_.has_value()), "flash_attentionXe35: FP8 does not support sinks");
  TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype, "flash_attentionXe35: q/k/v dtype mismatch");
  const bool has_all_descales = q_descale_.has_value() && k_descale_.has_value() && v_descale_.has_value();
  const bool has_any_descale = q_descale_.has_value() || k_descale_.has_value() || v_descale_.has_value();
  TORCH_CHECK(
      is_fp8 ? has_all_descales : !has_any_descale,
      "flash_attentionXe35: FP8 requires q/k/v descale; bf16 does not accept descale");

  float q_descale = 1.0f;
  float k_descale = 1.0f;
  float v_descale = 1.0f;
  if (is_fp8) {
    auto read_descale = [&](const at::Tensor& descale, const char* name) {
      TORCH_CHECK(descale.device().is_cpu(), "flash_attentionXe35: ", name, " must be a CPU scalar");
      TORCH_CHECK(descale.scalar_type() == at::ScalarType::Float, "flash_attentionXe35: ", name, " must be float32");
      TORCH_CHECK(descale.numel() == 1, "flash_attentionXe35: ", name, " must contain one per-tensor scale");
      return descale.item<float>();
    };
    q_descale = read_descale(q_descale_.value(), "q_descale");
    k_descale = read_descale(k_descale_.value(), "k_descale");
    v_descale = read_descale(v_descale_.value(), "v_descale");
  }
  TORCH_CHECK(q.is_xpu() && k.is_xpu() && v.is_xpu(), "flash_attentionXe35: q/k/v must be xpu tensors");
  TORCH_CHECK(cu_seqlens_q.is_xpu() && cu_seqlens_k.is_xpu(), "flash_attentionXe35: cu_seqlens must be xpu tensors");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: cache_seqlens must be int32");
  TORCH_CHECK(cu_seqlens_q.stride(0) == 1, "flash_attentionXe35: cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.stride(0) == 1, "flash_attentionXe35: cache_seqlens must be contiguous");
  TORCH_CHECK(
      q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
      "flash_attentionXe35: last dimension must be contiguous");

  const int total_q = static_cast<int>(q.size(0));
  const int num_heads_q = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  TORCH_CHECK(q.stride(1) == head_dim, "flash_attentionXe35: q must be contiguous on [total_q, h, d]");

  
  int num_heads_kv = 0;
  int page_size = 0;
  int num_pages_per_seq = 0;
  at::Tensor k_pool;
  at::Tensor v_pool;
  at::Tensor page_table_ref;

  if (is_paged) {
    page_table_ref = page_table.value();
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "flash_attentionXe35: k/v must be paged (num_pages, page_size, h_kv, d)");
    TORCH_CHECK(page_table_ref.dim() == 2, "flash_attentionXe35: page_table must be 2D");
    TORCH_CHECK(cu_seqlens_k.size(0) == batch, "flash_attentionXe35: cache_seqlens must have size equal to batch");
    TORCH_CHECK(page_table_ref.is_xpu(), "flash_attentionXe35: page_table must be an xpu tensor");
    TORCH_CHECK(page_table_ref.scalar_type() == at::ScalarType::Int, "flash_attentionXe35: page_table must be int32");
    TORCH_CHECK(page_table_ref.is_contiguous(), "flash_attentionXe35: page_table must be contiguous");

    const int num_pages = static_cast<int>(k.size(0));
    page_size = static_cast<int>(k.size(1));
    num_heads_kv = static_cast<int>(k.size(2));
    num_pages_per_seq = static_cast<int>(page_table_ref.size(1));
    TORCH_CHECK(page_size == 64 || page_size == 128, "flash_attentionXe35: only page_size 64 and 128 are supported");
    TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim, "flash_attentionXe35: head_dim mismatch between q/k/v");
    TORCH_CHECK(
        k.size(1) == v.size(1) && k.size(2) == v.size(2), "flash_attentionXe35: k/v shape mismatch on page/head axes");
    TORCH_CHECK(v.size(1) == page_size && v.size(2) == num_heads_kv, "flash_attentionXe35: k/v shape mismatch");

    k_pool = k.as_strided(
        {static_cast<int64_t>(num_pages) * page_size,
         static_cast<int64_t>(num_heads_kv),
         static_cast<int64_t>(head_dim)},
        {k.stride(1), k.stride(2), k.stride(3)});
    v_pool = v.as_strided(
        {static_cast<int64_t>(num_pages) * page_size,
         static_cast<int64_t>(num_heads_kv),
         static_cast<int64_t>(head_dim)},
        {v.stride(1), v.stride(2), v.stride(3)});
  } else {
    
    
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "flash_attentionXe35: non-paged k/v must be (total_k, h_kv, d)");
    TORCH_CHECK(
        cu_seqlens_k.size(0) == batch + 1, "flash_attentionXe35: cache_seqlens must have size equal to batch + 1");
    num_heads_kv = static_cast<int>(k.size(1));
    page_size = 0;
    num_pages_per_seq = 0;
    TORCH_CHECK(k.size(2) == head_dim && v.size(2) == head_dim, "flash_attentionXe35: head_dim mismatch between q/k/v");
    TORCH_CHECK(k.size(1) == v.size(1) && k.size(2) == v.size(2), "flash_attentionXe35: k/v shape mismatch");
    k_pool = k;
    v_pool = v;
    
    page_table_ref = at::empty({0}, q.options().dtype(at::kInt));
  }

  if (is_causal) {
    window_size_right = 0;
  }
  const bool is_local =
      (window_size_left >= 0 || window_size_right >= 0) && !(window_size_left < 0 && window_size_right == 0);
  const bool effective_causal = window_size_left < 0 && window_size_right == 0;
  if (window_size_left < 0) {
    window_size_left = max_seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = max_seqlen_q - 1;
  }

  const cutlass::bfloat16_t* sm_sink = nullptr;
  if (sinks_.has_value()) {
    const auto& sinks = sinks_.value();
    TORCH_CHECK(head_dim == 64, "flash_attentionXe35: sinks is only supported for head_dim 64");
    TORCH_CHECK(sinks.is_xpu() && sinks.device() == q.device(), "flash_attentionXe35: sinks must be on q's xpu device");
    TORCH_CHECK(sinks.scalar_type() == q_dtype, "flash_attentionXe35: sinks dtype must match q");
    TORCH_CHECK(
        sinks.dim() == 1 && sinks.size(0) == num_heads_q && sinks.stride(0) == 1,
        "flash_attentionXe35: sinks must be contiguous with shape [num_heads_q]");
    sm_sink = static_cast<const cutlass::bfloat16_t*>(sinks.data_ptr());
  }

  TORCH_CHECK(
      head_dim == 64 || head_dim == 128 || (is_paged && head_dim == 256),
      "flash_attentionXe35: non-paged attention supports head_dim 64 and 128; paged attention also supports 256");

  c10::DeviceGuard device_guard(q.device());
  auto opts = is_fp8 ? q.options().dtype(at::kBFloat16) : q.options();
  if (out_.has_value()) {
    const auto& out = out_.value();
    TORCH_CHECK(
        out.scalar_type() == opts.dtype().toScalarType(),
        "flash_attentionXe35: output dtype must match the kernel output dtype");
    TORCH_CHECK(
        out.dim() == 3 && out.size(0) == total_q && out.size(1) == num_heads_q && out.size(2) == head_dim,
        "flash_attentionXe35: output shape must be [total_q, num_heads_q, head_dim]");
    TORCH_CHECK(out.device() == q.device(), "flash_attentionXe35: output must be on the same device as q");
    TORCH_CHECK(out.stride(-1) == 1, "flash_attentionXe35: output must have a contiguous last dimension");
  }
  
  
  at::Tensor out_dense;
  if (out_.has_value()) {
    out_dense = out_.value();
  } else if (is_paged) {
    out_dense = at::empty({total_q, num_heads_q, head_dim}, opts);
  } else {
    out_dense = at::zeros({total_q, num_heads_q, head_dim}, opts);
  }

  auto dispatch = max_seqlen_q == 1 ? dispatch_xe35_decode_64_bf16_xe35 : dispatch_xe35_prefill_64_bf16_xe35;
  dispatch(
      q,
      k_pool,
      v_pool,
      page_table_ref,
      num_pages_per_seq,
      cu_seqlens_q,
      cu_seqlens_k,
      out_dense,
      batch,
      num_heads_q,
      num_heads_kv,
      max_seqlen_q,
      max_seqlen_k,
      page_size,
      q_descale,
      k_descale,
      v_descale,
      softmax_scale_,
      effective_causal,
      is_local,
      window_size_left,
      window_size_right,
      sm_sink,
      sinks_.has_value(),
      num_kv_splits);

  auto lse = at::zeros({num_heads_q, total_q}, opts.dtype(at::kFloat));
  auto out_accum = at::Tensor();
  auto lse_accum = at::Tensor();
  return {out_dense, lse, out_accum, lse_accum};
}

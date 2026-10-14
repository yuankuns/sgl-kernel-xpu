/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP4 x MXFP4 (W4A4) MoE grouped GEMM for Xe3.5 (CRI), ported from
           vllm-xpu-kernels @ 48313ef (csrc/xpu/grouped_gemm/xe_3/).

    Uses vLLM's MainloopMXFPXGroup plus its per-shape tile family and wave-quantization
    dispatch (see sycl/kernels/moe/xe35/collective/) instead of CUTLASS's in-tree
    MainloopIntelXeXMX16BlockScaledGroup with its single fixed 512x512x64 tile. On the
    muse_spark L3 MoE shapes (256 experts, K=4096) the vLLM collective is ~20-40x faster.

    Layout contract (identical to vLLM's cutlass_grouped_gemm_interface, because the
    strides are rebuilt from {N, K, groups} and the tensors' own strides are never read):
      a         [total_M, K/2]        float4_e2m1fn_x2, row-major
      a_scale   [padded_M, K/32]      float8_e8m0fnu, M-dense, each expert's M rounded up to 4
      b         [E, N, K/2]           float4_e2m1fn_x2, contiguous (i.e. K-dense / ColumnMajor)
      b_scale   contiguous [E, K/32, N] float8_e8m0fnu, or its
                [E, N, K/32] transposed view; N-dense in storage
      out       [total_M, N]          bfloat16
      rows_per_expert [E]             int32, on device
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdlib>

#include "sycl/kernels/moe/xe35/device/grouped_gemm_runner.hpp"

namespace gpu::cutlass_kernel::grouped_gemm {

// Small 256-expert MXFP4 projections benefit from a wider N tile and a
// two-stage pipeline while their average rows per expert remain in the decode
// regime.
class Mxfp4DecodeN256K64Stages2Policy : public moe_mxfp4_policy {
 public:
  using TileShape = Shape<_8, _256, _64>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>, Layout<TileShape>, SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup<PipelineStages>;
  CALL_GENERATE_GEMM();
};

}  // namespace gpu::cutlass_kernel::grouped_gemm

namespace {

using namespace sgl_kernel::vllm_grouped_gemm;
namespace policies = gpu::cutlass_kernel::grouped_gemm;

// The Xe 2D block load of the A-scales is DWord-granular, so the mainloop indexes each
// expert's block as ceil(K/32) * round_up(M_e, kScaleAlignRows) bytes.
constexpr int64_t kScaleAlignRows = 4;

// a_scale must therefore hold sum(round_up(M_e, kScaleAlignRows)) rows, which exceeds
// sum(M_e) by up to kScaleAlignRows - 1 per expert unless the caller already padded
// every expert's row block (sglang's `_xpu_w4a4_row_alignment` does). Getting this
// wrong means the mainloop reads past the end of the scale surface, so check it against
// the row counts themselves and not just against the shapes: that costs one
// num_experts-element D2H copy, i.e. a queue drain, per call. That drain is neither
// graph-recordable (the level-zero backend rejects a D2H copy inside an XPU graph with
// UR_RESULT_ERROR_UNSUPPORTED_FEATURE) nor safe to do between a rank's collectives (it
// waits on all-reduces the peers have not matched yet, so the ranks deadlock), so the
// row-count part is opt-in via SGL_KERNEL_MXFP4_CHECK_ROW_COUNTS=1. Normal
// callers must size a_scale for their per-expert padding contract.
bool row_count_checks_enabled() {
  const char* v = std::getenv("SGL_KERNEL_MXFP4_CHECK_ROW_COUNTS");
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}
void check_a_scale_rows(const torch::Tensor& rows_per_expert, int64_t a_scale_rows, int64_t total_m) {
  TORCH_CHECK(
      a_scale_rows >= total_m,
      "a_scale must have at least as many rows as a (",
      total_m,
      "), since it holds sum(round_up(M_e, ",
      kScaleAlignRows,
      ")) >= sum(M_e) rows, got ",
      a_scale_rows);

  // The shape-only check above is free and always runs; the row-count check needs the
  // device counts on the host, so it is opt-in (see row_count_checks_enabled).
  if (!row_count_checks_enabled()) {
    return;
  }

  const auto rows_cpu = rows_per_expert.to(at::kCPU).contiguous();
  const int32_t* rows = rows_cpu.data_ptr<int32_t>();
  int64_t total = 0;
  int64_t padded_total = 0;
  for (int64_t e = 0; e < rows_cpu.numel(); ++e) {
    TORCH_CHECK(rows[e] >= 0, "rows_per_expert[", e, "] is negative (", rows[e], ")");
    total += rows[e];
    padded_total += (rows[e] + kScaleAlignRows - 1) / kScaleAlignRows * kScaleAlignRows;
  }
  TORCH_CHECK(total <= total_m, "rows_per_expert sums to ", total, " rows but a only has ", total_m);
  TORCH_CHECK(
      padded_total <= a_scale_rows,
      "a_scale needs sum(round_up(M_e, ",
      kScaleAlignRows,
      ")) = ",
      padded_total,
      " rows for these row counts (sum(M_e) = ",
      total,
      "), but it has only ",
      a_scale_rows,
      ". Pad every expert's row block to a multiple of ",
      kScaleAlignRows,
      ", or size a_scale for the padding");
}

// Selects between the two general prefill tiles.
bool use_256x256_prefill_tile(int64_t M_total, int64_t N, int64_t groups) {
  constexpr int kCores = 32;
  constexpr double kMinUtil = 0.85;
  const int64_t M_g = M_total / (groups > 0 ? groups : 1);
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t tiles_256 = groups * ceil_div(M_g > 0 ? M_g : 1, 256) * ceil_div(N, 256);
  int64_t waves_256 = ceil_div(tiles_256, kCores);
  double util_256 = static_cast<double>(tiles_256) / static_cast<double>(waves_256 * kCores);

  return util_256 >= kMinUtil;
}

bool use_muse_spark_small_row_tile(int64_t M_total, int64_t N, int64_t K, int64_t groups) {
  // CRI operator sweeps with three routing histograms found a reliable
  // advantage through these allocated-row bounds. Down was near a tie at
  // 2944 rows for one histogram; gate/up became inconsistent at 3848.
  return groups == 256 && K == 4096 && ((N == 4096 && M_total <= 2880) || (N == 8192 && M_total <= 3840));
}

}  // namespace

void moe_grouped_mm_nt_xe35_mxfp4_w4a4(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& b,
    const torch::Tensor& b_scale,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& rows_per_expert,
    const int64_t n,
    const int64_t k,
    const int64_t num_experts) {
  auto& dpcpp_queue = at::xpu::getCurrentXPUStream(a.device().index()).queue();

  const int64_t groups = num_experts;
  const int64_t N = n;
  const int64_t K = k;
  const int64_t total_m = a.size(0);
  const int64_t scale_k = (K + 31) / 32;

  TORCH_CHECK(groups > 0 && N > 0 && K > 0, "n, k and num_experts must be > 0");
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64 (4-bit packing and the 32-element scale block)");

  // The XE 2D block copies need each row of B and D to start on a 128-bit boundary,
  // and cutlass reports a violation by aborting the process rather than returning.
  TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32 so each 4-bit weight row is 128-bit aligned");

  TORCH_CHECK(
      a.device().is_xpu() && b.device().is_xpu() && a_scale.device().is_xpu() && b_scale.device().is_xpu() &&
          output.device().is_xpu() && rows_per_expert.device().is_xpu(),
      "all tensors must be on an XPU device");

  // Packed 4-bit data and e8m0 scales are also commonly carried as uint8 in
  // sgl-kernel-xpu, so accept either the byte view or the typed one.
  auto is_packed_fp4 = [](const torch::Tensor& t) {
    return t.scalar_type() == at::kFloat4_e2m1fn_x2 || t.scalar_type() == at::kByte;
  };
  auto is_e8m0 = [](const torch::Tensor& t) {
    return t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte;
  };
  TORCH_CHECK(
      is_packed_fp4(a) && is_packed_fp4(b),
      "a and b must be float4_e2m1fn_x2 (or its uint8 view); got ",
      a.scalar_type(),
      " and ",
      b.scalar_type());
  TORCH_CHECK(
      is_e8m0(a_scale) && is_e8m0(b_scale),
      "a_scale and b_scale must be float8_e8m0fnu (or its uint8 view); got ",
      a_scale.scalar_type(),
      " and ",
      b_scale.scalar_type());
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");

  TORCH_CHECK(
      a.dim() == 2 && a.size(1) == K / 2 && a.is_contiguous(),
      "a must be a contiguous 2D [total_M, K/2] tensor of packed 4-bit activations");
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == total_m && output.size(1) == N && output.is_contiguous(),
      "output must be a contiguous 2D [total_M, N] tensor");
  TORCH_CHECK(
      b.dim() == 3 && b.size(0) == groups && b.size(1) == N && b.size(2) == K / 2 && b.is_contiguous(),
      "b must be a contiguous [num_experts, N, K/2] tensor");
  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.numel() == groups && rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be a 1D int32 tensor with one entry per expert");
  TORCH_CHECK(rows_per_expert.is_contiguous(), "rows_per_expert must be contiguous");

  // Only the extent of the A-scale surface is meaningful here: it is a plain
  // [padded_M, scale_k] buffer filled MN-major with each expert's row count rounded
  // up to a multiple of 4.
  TORCH_CHECK(
      a_scale.dim() == 2 && a_scale.size(1) == scale_k && a_scale.is_contiguous(),
      "a_scale must be a contiguous [padded_M, ceil(K/32)] tensor");
  // Callers that pad each expert's row block size the activation buffer from a host-side
  // upper bound rather than syncing the routing histogram, so a_scale is allowed to be
  // taller than the rows actually routed.
  check_a_scale_rows(rows_per_expert, a_scale.size(0), total_m);
  // Element (e, n, k) must sit at e * N * scale_k + n + k * N.
  TORCH_CHECK(
      b_scale.dim() == 3 && b_scale.size(0) == groups &&
          ((b_scale.size(1) == scale_k && b_scale.size(2) == N && b_scale.is_contiguous()) ||
           (b_scale.size(1) == N && b_scale.size(2) == scale_k && b_scale.stride(0) == N * scale_k &&
            (N == 1 || b_scale.stride(1) == 1) && (scale_k == 1 || b_scale.stride(2) == N))),
      "b_scale must be dense along N: either a contiguous [num_experts, ceil(K/32), N] tensor "
      "or its [num_experts, N, ceil(K/32)] transposed view");

  // The epilogue reads C as one fp32 row per token, so a per-expert bias must already
  // be expanded by the caller.
  if (bias) {
    TORCH_CHECK(
        bias->dim() == 2 && bias->size(0) == total_m && bias->size(1) == N && bias->is_contiguous() &&
            bias->scalar_type() == at::kFloat,
        "bias must be a contiguous fp32 [total_M, N] tensor");
  }

  const int64_t avg_tokens_cnt = total_m / groups;

#define SGL_GG_CALL(POLICY)              \
  kernel_functor<POLICY>(                \
      dpcpp_queue,                       \
      a.data_ptr(),                      \
      a_scale.data_ptr(),                \
      b.data_ptr(),                      \
      b_scale.data_ptr(),                \
      bias ? bias->data_ptr() : nullptr, \
      output.data_ptr(),                 \
      rows_per_expert.data_ptr(),        \
      N,                                 \
      K,                                 \
      groups)

  if (avg_tokens_cnt > 32) {
    if (K <= 1024 && N >= 1024) {
      // Narrow-K down-projection: a wider N tile amortizes the short K loop.
      if (avg_tokens_cnt <= 2048) {
        SGL_GG_CALL(policies::moe_mxfp4_256x128_policy);
      } else {
        SGL_GG_CALL(policies::moe_mxfp4_downproj_wide_policy);
      }
    } else {
      if (use_256x256_prefill_tile(total_m, N, groups)) {
        SGL_GG_CALL(policies::moe_mxfp4_policy);
      } else {
        SGL_GG_CALL(policies::moe_mxfp4_128x256_policy);
      }
    }
  } else if (use_muse_spark_small_row_tile(total_m, N, K, groups)) {
    SGL_GG_CALL(policies::Mxfp4DecodeN256K64Stages2Policy);
  } else if (avg_tokens_cnt > 4) {
    SGL_GG_CALL(policies::moe_mxfp4_mid_policy);
  } else {
    SGL_GG_CALL(policies::moe_mxfp4_decode_policy);
  }

#undef SGL_GG_CALL
}

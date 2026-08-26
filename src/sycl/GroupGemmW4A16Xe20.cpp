/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include <cstdlib>

#include "sgl_kernel_export.h"
#include "sycl/Utils.h"
#include "sycl/kernels/moe/xe20/w4a16/gemm_xe2_policy.hpp"
#ifdef USE_MOE_JIT
#include "jit/moe_jit.h"
#endif

namespace moe_w4a16 {
template <typename Policy, typename ElementS, typename ElementA>
void w4a16_launch(
    sycl::queue stream,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* zeros,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int* row_offsets,
    const int total_rows,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer);
}  // namespace moe_w4a16

#define DECLARE_W4A16_EXTERN(Policy, ElementS, ElementA)                               \
  extern template void moe_w4a16::w4a16_launch<moe_w4a16::Policy, ElementS, ElementA>( \
      sycl::queue,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      const void*,                                                                     \
      void*,                                                                           \
      const int,                                                                       \
      const int,                                                                       \
      const int*,                                                                      \
      const int*,                                                                      \
      const int,                                                                       \
      const int,                                                                       \
      const int,                                                                       \
      int32_t*);

#define DECLARE_W4A16_POLICY(Policy)                                     \
  DECLARE_W4A16_EXTERN(Policy, cutlass::bfloat16_t, cutlass::bfloat16_t) \
  DECLARE_W4A16_EXTERN(Policy, cutlass::half_t, cutlass::half_t)         \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::bfloat16_t)             \
  DECLARE_W4A16_EXTERN(Policy, uint8_t, cutlass::half_t)

DECLARE_W4A16_POLICY(w4a16_policy_m_8_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_16_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_32_n_64)
DECLARE_W4A16_POLICY(w4a16_policy_m_64_n_128)
DECLARE_W4A16_POLICY(w4a16_policy_m_64_n_256)
DECLARE_W4A16_POLICY(w4a16_policy_m_128_n_128)

#undef DECLARE_W4A16_POLICY
#undef DECLARE_W4A16_EXTERN

namespace {

// GEMM shape -> tile policy. Ids match the policy order in
// GroupGemmW4A16Xe20.cmake and w4a16_policy() in src/jit/moe_jit.cpp:
//   0: 8x64   1: 16x64   2: 32x64   3: 64x128   4: 64x256   5: 128x128
//
// The grouped GEMM tiles every expert's rows independently, so a policy with an
// M tile of T computes ceil(rows_e / T) * T rows for expert e, and a tile is
// only as good as the fraction of itself the rows fill. What the host can see is
// the *mean* rows per expert, and on real routing the mean does not bound that
// fraction: all three GPT-OSS-120B prefill layers average 128 rows per expert,
// yet their per-expert rows run from 0 to 3573, so a 128-row tile that the mean
// scores as a perfect fit really fills 0.61-0.82 of itself. Reading the routing
// histogram would need a device-to-host copy on every call.
//
// So the mean picks the M tile only where it bounds the padding -- below one
// tile of rows -- and everything above runs the 64-row tile, whose padding is at
// most ceil(rows_e / 64) * 64 whatever the tail looks like (1.234x on the real
// layer-0 histogram, against 1.6-1.9x for a 128-row tile). Measured on the real
// TP=4 layer-0 routing vector, the 64-row tile this picks beats the 128-row
// policy the menu used to select by 34% (gemm2: 60.5 against 45.1 TFLOP/s) and
// 36% (gemm1: 67.1 against 49.3).
//
// That leaves BLK_N, and the two 64-row tiles differ only there. Their
// padding-free rates on Arc Pro B60 (bf16 activations, gemm1 geometry with the M
// padding removed, TFLOP/s) are 82.9 for the 256-wide tile and 75.6 for the
// 128-wide one; only the ratio matters below, as it prices how much N tail the
// wider tile may waste before the narrower one wins.
constexpr float kW4A16PeakN256 = 82.9f;
constexpr float kW4A16PeakN128 = 75.6f;

// A short K demotes the wide tile outright. An MoE layer's second GEMM contracts
// over the sharded intermediate size (736 at TP=4, 384 at TP=8) instead of the
// hidden size, so its k-loop is 4-8x shorter and the per-work-group-tile costs
// the 256-wide tile pays -- its extra B load and its epilogue -- amortise over
// that much less dpas. Measured on the real TP=4 layer-0 gemm2 (N=2880, K=736)
// the 128-wide tile is 60.5 TFLOP/s against the 256-wide tile's 52.7, a 15% gap
// the N-fill score below cannot see: at N=2880 it scores the wide tile 77.7
// against 75.6 and would select it, so this branch has to come first.
constexpr int kW4A16ShortK = 1024;

int round_up(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

int select_w4a16_policy_id(int avg_m, int gemm_n, int gemm_k) {
  if (avg_m <= 4) return 0;
  if (avg_m <= 8) return 1;
  if (avg_m <= 32) return 2;
  if (gemm_k <= kW4A16ShortK) return 3;

  // Long K: the wider tile wins unless its N tail wastes more than it is worth.
  // A policy computes ceil(gemm_n / BLK_N) * BLK_N columns, and ceil at 256 is
  // never kinder than ceil at 128, so this only ever demotes the wide tile --
  // e.g. N=1152 leaves it 0.9 filled against 1.0, which its 1.097x rate cannot
  // pay for, while N=2880 (0.9375 against 1.0) it still wins on long K.
  const float fill_n256 = static_cast<float>(gemm_n) / static_cast<float>(round_up(gemm_n, 256));
  const float fill_n128 = static_cast<float>(gemm_n) / static_cast<float>(round_up(gemm_n, 128));
  return kW4A16PeakN256 * fill_n256 >= kW4A16PeakN128 * fill_n128 ? 4 : 3;
}

// Measurement hook: force one policy id so a benchmark can price every tile from
// a single build. Unset in production; an out-of-range value is ignored.
constexpr int kW4A16MaxPolicyId = 5;

int w4a16_forced_policy_id() {
  static const int forced = [] {
    const char* value = std::getenv("SGL_MOE_W4A16_POLICY_ID");
    if (value == nullptr || *value == '\0') return -1;
    const int id = std::atoi(value);
    return (id >= 0 && id <= kW4A16MaxPolicyId) ? id : -1;
  }();
  return forced;
}

}  // namespace

SGL_KERNEL_EXPORT void moe_grouped_mm_nt_xe20_w4a16(
    torch::Tensor& output,                   // [total_m, N] bf16/fp16
    const torch::Tensor& activations,        // [total_m, K] bf16 or fp16
    const torch::Tensor& packed_weights,     // [E, N, K/2] int8/uint8 (two 4-bit values per byte)
    const torch::Tensor& scales,             // [E, N, K/group_size]: int4=activation dtype, mxfp4=E8M0 byte
    const std::optional<at::Tensor>& zeros,  // [E, N, K/group_size], same dtype as int4 scales, optional
    const std::optional<at::Tensor>& bias,   // [E, N] float32, optional
    const torch::Tensor& rows_per_expert,    // [E] int32 per-expert row counts
    const int64_t n_experts,
    bool is_int4,
    const int64_t group_size) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(packed_weights);
  CHECK_INPUT(scales);
  CHECK_INPUT(rows_per_expert);
  TORCH_CHECK(output.device() == activations.device(), "output must be on the same device as activations");
  TORCH_CHECK(
      packed_weights.device() == activations.device(), "packed_weights must be on the same device as activations");
  TORCH_CHECK(scales.device() == activations.device(), "scales must be on the same device as activations");
  TORCH_CHECK(
      rows_per_expert.device() == activations.device(), "rows_per_expert must be on the same device as activations");
  if (zeros.has_value()) {
    const auto& zeros_tensor = *zeros;
    CHECK_INPUT(zeros_tensor);
    TORCH_CHECK(zeros_tensor.device() == activations.device(), "zeros must be on the same device as activations");
  }
  if (bias.has_value()) {
    const auto& bias_tensor = *bias;
    CHECK_INPUT(bias_tensor);
    TORCH_CHECK(bias_tensor.device() == activations.device(), "bias must be on the same device as activations");
  }

  TORCH_CHECK(output.dim() == 2, "output must be 2D [total_m, N]");
  TORCH_CHECK(activations.dim() == 2, "activations must be 2D [total_m, K]");
  TORCH_CHECK(rows_per_expert.dim() == 1, "rows_per_expert must be 1D [E]");
  const int total_m = activations.size(0);
  const int gemm_k = activations.size(1);

  auto pw_shape = packed_weights.sizes().vec();
  TORCH_CHECK(pw_shape.size() == 3, "packed_weights must be 3D [E, N, K/2]");
  const int gemm_n = pw_shape[1];
  TORCH_CHECK(pw_shape[0] == n_experts, "packed_weights.size(0) must equal n_experts");
  TORCH_CHECK(pw_shape[2] == gemm_k / 2, "packed_weights.size(2) must equal K/2 (two 4-bit values per byte)");
  TORCH_CHECK(
      packed_weights.scalar_type() == at::ScalarType::Char || packed_weights.scalar_type() == at::ScalarType::Byte,
      "packed_weights must be int8 or uint8");

  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256,
      "group_size must be 32, 64, 128 or 256; got ",
      group_size);
  TORCH_CHECK(gemm_k % group_size == 0, "K must be a multiple of group_size");

  auto sc_shape = scales.sizes().vec();
  TORCH_CHECK(sc_shape.size() == 3, "scales must be 3D [E, N, K/group_size]");
  TORCH_CHECK(sc_shape[0] == n_experts, "scales.size(0) must equal n_experts");
  TORCH_CHECK(sc_shape[1] == gemm_n, "scales.size(1) must equal N");
  TORCH_CHECK(sc_shape[2] == gemm_k / group_size, "scales.size(2) must equal K/group_size");
  if (is_int4) {
    TORCH_CHECK(scales.scalar_type() == activations.scalar_type(), "int4 scales dtype must match activations dtype");
  } else {
    TORCH_CHECK(
        scales.scalar_type() == at::ScalarType::Byte || scales.scalar_type() == at::ScalarType::Float8_e8m0fnu,
        "mxfp4 scales must be uint8 or float8_e8m0fnu (E8M0 exponent)");
  }

  TORCH_CHECK(n_experts > 0, "n_experts must be positive");
  TORCH_CHECK(n_experts == rows_per_expert.size(0), "rows_per_expert must have n_experts elements");
  TORCH_CHECK(rows_per_expert.scalar_type() == at::ScalarType::Int, "rows_per_expert must be int32");
  TORCH_CHECK(output.size(0) == total_m, "output rows must match activations rows");
  TORCH_CHECK(output.size(1) == gemm_n, "output must have N columns");
  TORCH_CHECK(gemm_n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16 || activations.scalar_type() == at::ScalarType::Half,
      "activations must be bfloat16 or half");
  TORCH_CHECK(output.scalar_type() == activations.scalar_type(), "output dtype must match activations dtype");

  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape must be [E, N]");
    bias_ptr = bias->data_ptr();
  }

  const void* zeros_ptr = nullptr;
  if (zeros.has_value()) {
    TORCH_CHECK(is_int4, "zeros (explicit zero-point) is only supported for int4, not mxfp4");
    TORCH_CHECK(zeros->scalar_type() == scales.scalar_type(), "zeros dtype must match int4 scales dtype");
    auto z_shape = zeros->sizes().vec();
    TORCH_CHECK(z_shape.size() == 3, "zeros must be 3D [E, N, K/group_size]");
    TORCH_CHECK(
        z_shape[0] == n_experts && z_shape[1] == gemm_n && z_shape[2] == gemm_k / group_size,
        "zeros shape must match scales shape [E, N, K/group_size]");
    zeros_ptr = zeros->data_ptr();
  }

  c10::DeviceGuard device_guard(activations.device());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  queue.memset(atomic_buffer.data_ptr<int>(), 0, sizeof(int32_t));

  const int avg_m = total_m / static_cast<int>(n_experts);
  const int forced_policy_id = w4a16_forced_policy_id();
  const int policy_id = forced_policy_id >= 0 ? forced_policy_id : select_w4a16_policy_id(avg_m, gemm_n, gemm_k);
  const bool is_fp16_act = activations.scalar_type() == at::ScalarType::Half;
#define LAUNCH_W4A16(Policy)                                                                  \
  do {                                                                                        \
    if (is_int4) {                                                                            \
      if (is_fp16_act) {                                                                      \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::half_t, cutlass::half_t>(         \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            nullptr,                                                                          \
            total_m,                                                                          \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      } else {                                                                                \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, cutlass::bfloat16_t, cutlass::bfloat16_t>( \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            nullptr,                                                                          \
            total_m,                                                                          \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      }                                                                                       \
    } else {                                                                                  \
      if (is_fp16_act) {                                                                      \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::half_t>(                 \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            nullptr,                                                                          \
            total_m,                                                                          \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      } else {                                                                                \
        moe_w4a16::w4a16_launch<moe_w4a16::Policy, uint8_t, cutlass::bfloat16_t>(             \
            queue,                                                                            \
            activations.data_ptr(),                                                           \
            packed_weights.data_ptr(),                                                        \
            scales.data_ptr(),                                                                \
            zeros_ptr,                                                                        \
            bias_ptr,                                                                         \
            output.data_ptr(),                                                                \
            gemm_n,                                                                           \
            gemm_k,                                                                           \
            rows_per_expert.data_ptr<int>(),                                                  \
            nullptr,                                                                          \
            total_m,                                                                          \
            static_cast<int>(n_experts),                                                      \
            static_cast<int>(group_size),                                                     \
            atomic_buffer.data_ptr<int>());                                                   \
      }                                                                                       \
    }                                                                                         \
  } while (0)

#define DISPATCH_W4A16_POLICY()                 \
  do {                                          \
    switch (policy_id) {                        \
      case 0:                                   \
        LAUNCH_W4A16(w4a16_policy_m_8_n_64);    \
        break;                                  \
      case 1:                                   \
        LAUNCH_W4A16(w4a16_policy_m_16_n_64);   \
        break;                                  \
      case 2:                                   \
        LAUNCH_W4A16(w4a16_policy_m_32_n_64);   \
        break;                                  \
      case 3:                                   \
        LAUNCH_W4A16(w4a16_policy_m_64_n_128);  \
        break;                                  \
      case 4:                                   \
        LAUNCH_W4A16(w4a16_policy_m_64_n_256);  \
        break;                                  \
      case 5:                                   \
        LAUNCH_W4A16(w4a16_policy_m_128_n_128); \
        break;                                  \
    }                                           \
  } while (0)

#ifdef USE_MOE_JIT
  {
    std::string jit_err;
    TORCH_CHECK(
        sgl::moe_jit::w4a16_grouped_gemm_launch(
            policy_id,
            is_int4,
            is_fp16_act,
            &queue,
            activations.data_ptr(),
            packed_weights.data_ptr(),
            scales.data_ptr(),
            zeros_ptr,
            bias_ptr,
            output.data_ptr(),
            gemm_n,
            gemm_k,
            rows_per_expert.data_ptr<int>(),
            nullptr,
            total_m,
            static_cast<int>(n_experts),
            static_cast<int>(group_size),
            atomic_buffer.data_ptr<int>(),
            jit_arch_code(),
            &jit_err),
        jit_err);
  }
#else
  DISPATCH_W4A16_POLICY();
#endif

#undef DISPATCH_W4A16_POLICY
#undef LAUNCH_W4A16
}

#undef SYCL_INTEL_TARGET

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"

namespace at::native::xpu {
namespace {

constexpr int kVecSize = 8;
using fp32_vec_t = aligned_vector_loop<float, kVecSize>;

template <int HiddenSize, bool HasPostWeight, bool PostNormInputScale, bool RoundPreNormInput, bool HasOutputFp32>
struct FusedPostNormResidualPreNorm {
  // One 8-element vector per lane through H=8192; larger supported rows
  // reuse the same 1024-lane workgroup with two vectors per lane.
  static constexpr int kWorkGroupSize = HiddenSize < 8192 ? HiddenSize / kVecSize : 1024;
  static constexpr int kIters = HiddenSize / (kWorkGroupSize * kVecSize);
  using bf16_vec_t = aligned_vector_loop<at::BFloat16, kVecSize>;

  const at::BFloat16* __restrict__ branch;
  const float* __restrict__ stream;
  const at::BFloat16* __restrict__ post_weight;
  const at::BFloat16* __restrict__ pre_weight;
  const float* __restrict__ alpha;
  const float* __restrict__ beta;
  at::BFloat16* __restrict__ output;
  float* __restrict__ new_stream;
  float* __restrict__ output_fp32;
  float post_eps;
  float pre_eps;

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    const int lane = item.get_local_id(0);
    const int row_offset = row * HiddenSize;

    bf16_vec_t branch_reg[kIters];
    float branch_sum_sq = 0.0f;

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int offset = (it * kWorkGroupSize + lane) * kVecSize;
      const bf16_vec_t value = *reinterpret_cast<const bf16_vec_t*>(branch + row_offset + offset);
      branch_reg[it] = value;
#pragma unroll
      for (int v = 0; v < kVecSize; ++v) {
        float x = static_cast<float>(value[v]);
        if constexpr (HasPostWeight && PostNormInputScale) {
          x *= static_cast<float>(post_weight[offset + v]);
        }
        branch_sum_sq += x * x;
      }
    }

    branch_sum_sq = sycl::reduce_over_group(item.get_group(), branch_sum_sq, sycl::plus<float>());
    const float post_rstd = sycl::native::rsqrt(branch_sum_sq / static_cast<float>(HiddenSize) + post_eps);

    fp32_vec_t new_stream_reg[kIters];
    float stream_sum_sq = 0.0f;

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int offset = (it * kWorkGroupSize + lane) * kVecSize;
      const fp32_vec_t stream_value = *reinterpret_cast<const fp32_vec_t*>(stream + row_offset + offset);
      const fp32_vec_t alpha_value = *reinterpret_cast<const fp32_vec_t*>(alpha + offset);
      const fp32_vec_t beta_value = *reinterpret_cast<const fp32_vec_t*>(beta + offset);
      fp32_vec_t updated;

#pragma unroll
      for (int v = 0; v < kVecSize; ++v) {
        // Match the model's BF16 post-norm output before the FP32 residual add.
        float post_norm_value = static_cast<float>(branch_reg[it][v]);
        if constexpr (HasPostWeight && PostNormInputScale) {
          post_norm_value *= static_cast<float>(post_weight[offset + v]);
        }
        post_norm_value *= post_rstd;
        if constexpr (HasPostWeight && !PostNormInputScale) {
          post_norm_value *= static_cast<float>(post_weight[offset + v]);
        }
        const at::BFloat16 normed_bf16 = static_cast<at::BFloat16>(post_norm_value);
        const float value = alpha_value[v] * stream_value[v] + beta_value[v] * static_cast<float>(normed_bf16);
        updated[v] = value;
        const float pre_norm_value = RoundPreNormInput ? static_cast<float>(static_cast<at::BFloat16>(value)) : value;
        stream_sum_sq += pre_norm_value * pre_norm_value;
      }
      new_stream_reg[it] = updated;
    }

    stream_sum_sq = sycl::reduce_over_group(item.get_group(), stream_sum_sq, sycl::plus<float>());
    const float pre_rstd = sycl::native::rsqrt(stream_sum_sq / static_cast<float>(HiddenSize) + pre_eps);

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int offset = (it * kWorkGroupSize + lane) * kVecSize;
      const bf16_vec_t weight = *reinterpret_cast<const bf16_vec_t*>(pre_weight + offset);
      bf16_vec_t normalized;
#pragma unroll
      for (int v = 0; v < kVecSize; ++v) {
        const float pre_norm_value = RoundPreNormInput
                                         ? static_cast<float>(static_cast<at::BFloat16>(new_stream_reg[it][v]))
                                         : new_stream_reg[it][v];
        normalized[v] = static_cast<at::BFloat16>(pre_norm_value * pre_rstd * static_cast<float>(weight[v]));
      }
      *reinterpret_cast<fp32_vec_t*>(new_stream + row_offset + offset) = new_stream_reg[it];
      *reinterpret_cast<bf16_vec_t*>(output + row_offset + offset) = normalized;
      if constexpr (HasOutputFp32) {
        fp32_vec_t widened;
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          widened[v] = static_cast<float>(normalized[v]);
        }
        *reinterpret_cast<fp32_vec_t*>(output_fp32 + row_offset + offset) = widened;
      }
    }
  }
};

template <int HiddenSize, bool HasPostWeight, bool PostNormInputScale, bool RoundPreNormInput, bool HasOutputFp32>
void launch_fused_post_norm_residual_pre_norm(
    int64_t rows,
    const at::BFloat16* branch,
    const float* stream,
    const at::BFloat16* post_weight,
    const at::BFloat16* pre_weight,
    const float* alpha,
    const float* beta,
    at::BFloat16* output,
    float* new_stream,
    float* output_fp32,
    float post_eps,
    float pre_eps,
    sycl::queue& queue) {
  FusedPostNormResidualPreNorm<HiddenSize, HasPostWeight, PostNormInputScale, RoundPreNormInput, HasOutputFp32> kernel{
      branch, stream, post_weight, pre_weight, alpha, beta, output, new_stream, output_fp32, post_eps, pre_eps};
  constexpr int kWorkGroupSize =
      FusedPostNormResidualPreNorm<HiddenSize, HasPostWeight, PostNormInputScale, RoundPreNormInput, HasOutputFp32>::
          kWorkGroupSize;
  sycl_kernel_submit(sycl::range<1>(rows * kWorkGroupSize), sycl::range<1>(kWorkGroupSize), queue, kernel);
}

}  // namespace

void fused_post_norm_residual_pre_norm(
    torch::Tensor branch,
    torch::Tensor stream,
    torch::Tensor post_weight,
    torch::Tensor pre_weight,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor new_stream,
    double post_eps,
    double pre_eps,
    bool post_norm_input_scale,
    bool round_pre_norm_input,
    const std::optional<torch::Tensor>& output_fp32) {
  TORCH_CHECK(branch.device().is_xpu(), "branch must be an XPU tensor");
  TORCH_CHECK(branch.scalar_type() == at::kBFloat16, "branch must be bfloat16");
  TORCH_CHECK(stream.scalar_type() == at::kFloat, "stream must be float32");
  TORCH_CHECK(post_weight.scalar_type() == at::kBFloat16, "post_weight must be bfloat16");
  TORCH_CHECK(pre_weight.scalar_type() == at::kBFloat16, "pre_weight must be bfloat16");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(new_stream.scalar_type() == at::kFloat, "new_stream must be float32");
  TORCH_CHECK(
      branch.dim() == 2 && (branch.size(1) == 4096 || branch.size(1) == 8192 || branch.size(1) == 16384),
      "branch must have shape [M, H] with H=4096, 8192, or 16384");
  const int64_t hidden_size = branch.size(1);
  TORCH_CHECK(stream.sizes() == branch.sizes(), "stream must match branch shape");
  TORCH_CHECK(output.sizes() == branch.sizes(), "output must match branch shape");
  TORCH_CHECK(new_stream.sizes() == branch.sizes(), "new_stream must match branch shape");
  TORCH_CHECK(post_weight.numel() == 0 || post_weight.numel() == hidden_size, "post_weight must be empty or match H");
  TORCH_CHECK(pre_weight.numel() == hidden_size, "pre_weight must match H");
  TORCH_CHECK(alpha.numel() == hidden_size, "alpha must match H");
  TORCH_CHECK(beta.numel() == hidden_size, "beta must match H");
  TORCH_CHECK(branch.is_contiguous(), "branch must be contiguous");
  TORCH_CHECK(stream.is_contiguous(), "stream must be contiguous");
  TORCH_CHECK(post_weight.is_contiguous(), "post_weight must be contiguous");
  TORCH_CHECK(pre_weight.is_contiguous(), "pre_weight must be contiguous");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(new_stream.is_contiguous(), "new_stream must be contiguous");
  // Optional FP32 widening of `output` for consumers with FP32 weights (the MoE
  // router), saving them a separate cast launch.
  float* output_fp32_ptr = nullptr;
  if (output_fp32.has_value()) {
    TORCH_CHECK(output_fp32->scalar_type() == at::kFloat, "output_fp32 must be float32");
    TORCH_CHECK(output_fp32->sizes() == branch.sizes(), "output_fp32 must match branch shape");
    TORCH_CHECK(output_fp32->is_contiguous(), "output_fp32 must be contiguous");
    output_fp32_ptr = output_fp32->data_ptr<float>();
  }

  const int64_t rows = branch.size(0);
  if (rows == 0) {
    return;
  }

  auto queue = at::xpu::getCurrentXPUStream().queue();
#define LAUNCH_KERNEL(HIDDEN_SIZE, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT, HAS_OUTPUT_FP32) \
  launch_fused_post_norm_residual_pre_norm<                                                                       \
      HIDDEN_SIZE,                                                                                                \
      HAS_POST_WEIGHT,                                                                                            \
      POST_NORM_INPUT_SCALE,                                                                                      \
      ROUND_PRE_NORM_INPUT,                                                                                       \
      HAS_OUTPUT_FP32>(                                                                                           \
      rows,                                                                                                       \
      branch.data_ptr<at::BFloat16>(),                                                                            \
      stream.data_ptr<float>(),                                                                                   \
      post_weight.data_ptr<at::BFloat16>(),                                                                       \
      pre_weight.data_ptr<at::BFloat16>(),                                                                        \
      alpha.data_ptr<float>(),                                                                                    \
      beta.data_ptr<float>(),                                                                                     \
      output.data_ptr<at::BFloat16>(),                                                                            \
      new_stream.data_ptr<float>(),                                                                               \
      output_fp32_ptr,                                                                                            \
      static_cast<float>(post_eps),                                                                               \
      static_cast<float>(pre_eps),                                                                                \
      queue)
#define LAUNCH_FOR_H(HIDDEN_SIZE, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT)      \
  if (output_fp32_ptr != nullptr) {                                                                  \
    LAUNCH_KERNEL(HIDDEN_SIZE, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT, true);  \
  } else {                                                                                           \
    LAUNCH_KERNEL(HIDDEN_SIZE, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT, false); \
  }
#define LAUNCH(HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT)           \
  if (hidden_size == 4096) {                                                           \
    LAUNCH_FOR_H(4096, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT);  \
  } else if (hidden_size == 8192) {                                                    \
    LAUNCH_FOR_H(8192, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT);  \
  } else {                                                                             \
    LAUNCH_FOR_H(16384, HAS_POST_WEIGHT, POST_NORM_INPUT_SCALE, ROUND_PRE_NORM_INPUT); \
  }
  const bool has_post_weight = post_weight.numel() != 0;
  if (has_post_weight && post_norm_input_scale && round_pre_norm_input) {
    LAUNCH(true, true, true);
  } else if (has_post_weight && post_norm_input_scale) {
    LAUNCH(true, true, false);
  } else if (has_post_weight && round_pre_norm_input) {
    LAUNCH(true, false, true);
  } else if (has_post_weight) {
    LAUNCH(true, false, false);
  } else if (round_pre_norm_input) {
    LAUNCH(false, false, true);
  } else {
    LAUNCH(false, false, false);
  }
#undef LAUNCH
#undef LAUNCH_FOR_H
#undef LAUNCH_KERNEL
}

}  // namespace at::native::xpu

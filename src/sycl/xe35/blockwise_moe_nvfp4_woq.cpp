/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief NVFP4 weight-only MoE GEMMs for Xe3P.

    The MXFP8 and BF16 compute paths share the same policy builder, persistent
    expert scheduler, contiguous tensor mapping, epilogue, and launcher as the
    MXFP4 MoE implementation. Only the fused NVFP4 conversion mainloops remain
    format-specific.
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdlib>
#include <cstring>
#include <cute/tensor.hpp>

#include "nvfp4_woq_bf16_mainloop.hpp"
#include "nvfp4_woq_mxfp8_mainloop.hpp"
#include "sycl/kernels/moe/xe35/device/grouped_gemm_runner.hpp"

using namespace cute;

void mxfp4_moe_scale_to_mn_major(torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert);

namespace gpu::cutlass_kernel::grouped_gemm {

template <
    bool QuantizeInKLoop,
    sgl_xpu::NvFp4WideCache CacheMode,
    class TileShape_,
    class SGLayout_,
    int PipelineStages_ = 2>
class moe_nvfp4_mxfp8_policy {
 public:
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementScaleA = typename cutlass::mx_float8_t<cutlass::float_e4m3_t>::ScaleFactorType;
  using ElementScaleB = ElementScaleA;
  using ElementAccumulator = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using StrideScale = cute::Stride<cute::_1, int64_t, int64_t>;

  using TileShape = TileShape_;
  using SGLayout = SGLayout_;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, ElementAccumulator, ElementA>>, Layout<TileShape>, SGLayout>::
          TiledMMA;

  static constexpr int PipelineStages = PipelineStages_;
  using GEMMDispatchPolicy = sgl_xpu::MainloopXeNvFp4WoqMxFp8Group<PipelineStages, QuantizeInKLoop, CacheMode>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementA, ElementScaleA>,
      cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA>, StrideScale>,
      cute::tuple<ElementB, ElementScaleB>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB>, StrideScale>,
      TiledMma,
      cute::type_list<void, void>,
      void,
      void,
      cute::identity,
      cute::type_list<void, void>,
      void,
      void,
      cute::identity>;

  using EpilogueDispatchPolicy = cutlass::epilogue::MoE16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      float,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using GemmBuilder = GenerateGemmFromCollective<
      CollectiveMainloop,
      ElementAccumulator,
      ElementOutput,
      LayoutC,
      LayoutD,
      TiledMma,
      TileShape,
      EpilogueDispatchPolicy,
      EpilogueOp>;
  using Gemm = typename GemmBuilder::Gemm;
};

template <bool QuantizeInKLoop, sgl_xpu::NvFp4WideCache CacheMode = sgl_xpu::NvFp4WideCache::None>
using moe_nvfp4_mxfp8_m8n128_policy = moe_nvfp4_mxfp8_policy<
    QuantizeInKLoop,
    CacheMode,
    Shape<_8, _128, _64>,
    Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>,
    4>;

template <bool QuantizeInKLoop, sgl_xpu::NvFp4WideCache CacheMode = sgl_xpu::NvFp4WideCache::None>
using moe_nvfp4_mxfp8_m8n256_policy = moe_nvfp4_mxfp8_policy<
    QuantizeInKLoop,
    CacheMode,
    Shape<_8, _256, _64>,
    Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>,
    4>;

template <bool QuantizeInKLoop, sgl_xpu::NvFp4WideCache CacheMode = sgl_xpu::NvFp4WideCache::None>
using moe_nvfp4_mxfp8_m32n256_policy = moe_nvfp4_mxfp8_policy<
    QuantizeInKLoop,
    CacheMode,
    Shape<_32, _256, _64>,
    Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>;

template <bool QuantizeInKLoop, sgl_xpu::NvFp4WideCache CacheMode = sgl_xpu::NvFp4WideCache::None>
using moe_nvfp4_mxfp8_m64n256_policy = moe_nvfp4_mxfp8_policy<
    QuantizeInKLoop,
    CacheMode,
    Shape<_64, _256, _64>,
    Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>;

template <bool QuantizeInKLoop, sgl_xpu::NvFp4WideCache CacheMode = sgl_xpu::NvFp4WideCache::None>
using moe_nvfp4_mxfp8_m256n256_policy = moe_nvfp4_mxfp8_policy<
    QuantizeInKLoop,
    CacheMode,
    Shape<_256, _256, _64>,
    Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>;

template <class TileShape_, class SGLayout_>
class moe_nvfp4_bf16_policy {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementAccumulator = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = TileShape_;
  using SGLayout = SGLayout_;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>, Layout<TileShape>, SGLayout>::
          TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = sgl_xpu::MainloopXeNvFp4WoqBf16Group<PipelineStages>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;

  using EpilogueDispatchPolicy = cutlass::epilogue::MoE16Group;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      float,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using GemmBuilder = GenerateGemmFromCollective<
      CollectiveMainloop,
      ElementAccumulator,
      ElementOutput,
      LayoutC,
      LayoutD,
      TiledMma,
      TileShape,
      EpilogueDispatchPolicy,
      EpilogueOp>;
  using Gemm = typename GemmBuilder::Gemm;
};

using moe_nvfp4_bf16_m8n128_policy =
    moe_nvfp4_bf16_policy<Shape<_8, _128, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>;
using moe_nvfp4_bf16_m8n256_policy =
    moe_nvfp4_bf16_policy<Shape<_8, _256, _32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>;
using moe_nvfp4_bf16_m32n256_policy =
    moe_nvfp4_bf16_policy<Shape<_32, _256, _32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>;
using moe_nvfp4_bf16_m64n256_policy =
    moe_nvfp4_bf16_policy<Shape<_64, _256, _32>, Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>>;
using moe_nvfp4_bf16_m256n256_policy =
    moe_nvfp4_bf16_policy<Shape<_256, _256, _32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>;

}  // namespace gpu::cutlass_kernel::grouped_gemm

namespace {

namespace policies = gpu::cutlass_kernel::grouped_gemm;
using sgl_kernel::vllm_grouped_gemm::GroupedGemmRunner;

constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kNvFp4BlockSize = 16;

bool nvfp4_woq_quantize_in_k_loop_enabled() {
  const char* value = std::getenv("SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0) {
    return false;
  }
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
      std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0) {
    return true;
  }
  TORCH_CHECK(
      false,
      "SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP must be one of 0/1, "
      "false/true, or off/on; got '",
      value,
      "'");
}

sgl_xpu::NvFp4WideCache nvfp4_woq_wide_cache_mode() {
  const char* value = std::getenv("SGLANG_NVFP4_WOQ_WIDE_CACHE");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "none") == 0 || std::strcmp(value, "register") == 0) {
    return sgl_xpu::NvFp4WideCache::None;
  }
  if (std::strcmp(value, "l1") == 0) {
    return sgl_xpu::NvFp4WideCache::L1;
  }
  if (std::strcmp(value, "slm") == 0) {
    return sgl_xpu::NvFp4WideCache::Slm;
  }
  TORCH_CHECK(
      false,
      "SGLANG_NVFP4_WOQ_WIDE_CACHE must be none, register, l1, or slm; "
      "got '",
      value,
      "'");
}

enum class MTile {
  M8N128,
  M8N256,
  M32N256,
  M64N256,
  M256N256,
};

MTile select_m_tile(int64_t total_rows, int64_t groups, int64_t n, int sm_count) {
  const char* forced = std::getenv("SGLANG_NVFP4_WOQ_FORCE_TILE");
  if (forced != nullptr && forced[0] != '\0' && std::strcmp(forced, "auto") != 0) {
    if (std::strcmp(forced, "m8n128") == 0) {
      return MTile::M8N128;
    }
    if (std::strcmp(forced, "m8n256") == 0) {
      return MTile::M8N256;
    }
    if (std::strcmp(forced, "m32n256") == 0) {
      return MTile::M32N256;
    }
    if (std::strcmp(forced, "m64n256") == 0) {
      return MTile::M64N256;
    }
    if (std::strcmp(forced, "m256n256") == 0) {
      return MTile::M256N256;
    }
    TORCH_CHECK(
        false,
        "SGLANG_NVFP4_WOQ_FORCE_TILE must be auto, m8n128, m8n256, "
        "m32n256, m64n256, or m256n256; got '",
        forced,
        "'");
  }

  const int64_t rows_per_group = (total_rows + groups - 1) / groups;
  if (rows_per_group <= 8) {
    const int64_t m8n256_workgroups = groups * cute::ceil_div(n, int64_t(256));
    return m8n256_workgroups < sm_count ? MTile::M8N128 : MTile::M8N256;
  }
  if (rows_per_group <= 32) {
    return MTile::M32N256;
  }
  if (rows_per_group <= 64) {
    return MTile::M64N256;
  }
  return MTile::M256N256;
}

cutlass::KernelHardwareInfo hardware_info(const torch::Device& device) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(device.index());
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  return hw_info;
}

template <class Policy>
void run_mxfp8_policy(
    sycl::queue& queue,
    const cutlass::KernelHardwareInfo& hw_info,
    const torch::Tensor& rows_per_expert,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& workspace,
    torch::Tensor& output,
    int64_t n,
    int64_t k,
    int64_t groups) {
  using Gemm = typename Policy::Gemm;
  using Mainloop = typename Gemm::CollectiveMainloop;
  GroupedGemmRunner<Gemm, false> runner;
  typename Mainloop::Arguments mainloop_args{
      reinterpret_cast<const typename Mainloop::ElementA*>(a.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementB*>(b.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementScaleA*>(scales_a.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementScaleB*>(sfmx_b.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementMult*>(c_b.data_ptr()),
      reinterpret_cast<typename Mainloop::ElementMult*>(workspace.data_ptr())};
  if constexpr (Mainloop::DispatchPolicy::CacheMode == sgl_xpu::NvFp4WideCache::L1) {
    constexpr int kMaxThreadsPerSM = 512;
    constexpr int kWorkgroupsPerSM = kMaxThreadsPerSM / Gemm::GemmKernel::MaxThreadsPerBlock;
    const int64_t required_bytes = static_cast<int64_t>(hw_info.sm_count) * kWorkgroupsPerSM * Mainloop::Base::BLK_N *
                                   Mainloop::Base::BLK_K * sizeof(typename Mainloop::ElementMult);
    TORCH_CHECK(
        workspace.nbytes() >= required_bytes,
        "nvfp4_woq: L1 widened cache requires at least ",
        required_bytes,
        " workspace bytes, got ",
        workspace.nbytes());
  }
  runner.run_with_mainloop(
      queue,
      hw_info,
      rows_per_expert.data_ptr<int32_t>(),
      mainloop_args,
      nullptr,
      output.data_ptr<float>(),
      n,
      k,
      groups);
}

template <class Policy>
void run_bf16_policy(
    sycl::queue& queue,
    const cutlass::KernelHardwareInfo& hw_info,
    const torch::Tensor& rows_per_expert,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    torch::Tensor& output,
    int64_t n,
    int64_t k,
    int64_t groups) {
  using Gemm = typename Policy::Gemm;
  using Mainloop = typename Gemm::CollectiveMainloop;
  GroupedGemmRunner<Gemm, false> runner;
  typename Mainloop::Arguments mainloop_args{
      reinterpret_cast<const typename Mainloop::ElementA*>(a.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementB*>(b.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementScale*>(sfmx_b.data_ptr()),
      reinterpret_cast<const typename Mainloop::ElementMult*>(c_b.data_ptr())};
  runner.run_with_mainloop(
      queue,
      hw_info,
      rows_per_expert.data_ptr<int32_t>(),
      mainloop_args,
      nullptr,
      output.data_ptr<float>(),
      n,
      k,
      groups);
}

void check_group_metadata(
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets, const torch::Device& device) {
  TORCH_CHECK(problem_sizes.device() == device, "nvfp4_woq: problem_sizes must be on the same XPU device as A");
  TORCH_CHECK(expert_offsets.device() == device, "nvfp4_woq: expert_offsets must be on the same XPU device as A");
  TORCH_CHECK(
      problem_sizes.dim() == 2 && problem_sizes.size(1) == 3 && problem_sizes.scalar_type() == torch::kInt32,
      "nvfp4_woq: problem_sizes must be (E, 3) int32");
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.scalar_type() == torch::kInt32 &&
          expert_offsets.numel() == problem_sizes.size(0),
      "nvfp4_woq: expert_offsets must be (E) int32");
}

}  // namespace

void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq: A must be on XPU");
  const auto device = a.device();
  TORCH_CHECK(
      b.device() == device && scales_a.device() == device && sfmx_b.device() == device && c_b.device() == device &&
          output.device() == device && workspace.device() == device,
      "nvfp4_woq: all tensors must be on the same XPU device");
  check_group_metadata(problem_sizes, expert_offsets, device);

  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "nvfp4_woq: A must be float8_e4m3fn (MXFP8)");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "nvfp4_woq: B must be uint8 (packed E2M1)");
  TORCH_CHECK(
      scales_a.scalar_type() == torch::kUInt8 && sfmx_b.scalar_type() == torch::kUInt8,
      "nvfp4_woq: activation and prepared weight scales must be uint8");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq: output must be float32");
  TORCH_CHECK(
      a.dim() == 2 && scales_a.dim() == 2 && output.dim() == 2 && b.dim() == 3 && sfmx_b.dim() == 3 && c_b.dim() == 3,
      "nvfp4_woq: invalid tensor ranks");
  TORCH_CHECK(
      a.is_contiguous() && b.is_contiguous() && scales_a.is_contiguous() && sfmx_b.is_contiguous() &&
          c_b.is_contiguous() && output.is_contiguous(),
      "nvfp4_woq: all data tensors must be contiguous");

  const int64_t groups = problem_sizes.size(0);
  const int64_t total_m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = output.size(1);
  TORCH_CHECK(groups > 0 && total_m > 0, "nvfp4_woq: E and total M must be positive");
  TORCH_CHECK(k % 64 == 0, "nvfp4_woq: K must be a multiple of 64");
  TORCH_CHECK(n % 4 == 0, "nvfp4_woq: N must be a multiple of 4");
  TORCH_CHECK(output.size(0) == total_m, "nvfp4_woq: A and output row counts must match");
  TORCH_CHECK(
      scales_a.size(0) == total_m && scales_a.size(1) == k / kMxBlockSize,
      "nvfp4_woq: scales_a must be (total_M, K/32)");
  TORCH_CHECK(b.size(0) == groups && b.size(1) == n && b.size(2) == k / 2, "nvfp4_woq: B must be (E, N, K/2)");
  TORCH_CHECK(
      sfmx_b.size(0) == groups && sfmx_b.size(1) == k / kMxBlockSize && sfmx_b.size(2) == n,
      "nvfp4_woq: sfmx_b must be (E, K/32, N)");
  TORCH_CHECK(
      sfmx_b.stride(0) % 64 == 0,
      "nvfp4_woq: N * (K/32) must be a multiple of 64 so each expert's "
      "B-scale slice is 64-byte aligned; got per-expert stride ",
      sfmx_b.stride(0));
  TORCH_CHECK(
      c_b.size(0) == groups && c_b.size(1) == n && c_b.size(2) == k / kNvFp4BlockSize,
      "nvfp4_woq: c_b must be (E, N, K/16)");

  auto rows_per_expert = problem_sizes.select(1, 0).contiguous();
  auto scales_a_mn = torch::zeros({total_m + 3 * groups, k / kMxBlockSize}, scales_a.options());
  mxfp4_moe_scale_to_mn_major(scales_a_mn, scales_a, rows_per_expert);

  auto& queue = at::xpu::getCurrentXPUStream(device.index()).queue();
  const auto hw_info = hardware_info(device);
  const auto tile = select_m_tile(total_m, groups, n, hw_info.sm_count);

#define RUN_MXFP8_POLICY(...)    \
  run_mxfp8_policy<__VA_ARGS__>( \
      queue, hw_info, rows_per_expert, a, b, scales_a_mn, sfmx_b, c_b, workspace, output, n, k, groups)

  const auto cache_mode = nvfp4_woq_wide_cache_mode();
  if (cache_mode == sgl_xpu::NvFp4WideCache::L1) {
    switch (tile) {
      case MTile::M8N128:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n128_policy<true>);
        break;
      case MTile::M8N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n256_policy<true>);
        break;
      case MTile::M32N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m32n256_policy<true, sgl_xpu::NvFp4WideCache::L1>);
        break;
      case MTile::M64N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m64n256_policy<true, sgl_xpu::NvFp4WideCache::L1>);
        break;
      case MTile::M256N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m256n256_policy<true, sgl_xpu::NvFp4WideCache::L1>);
        break;
    }
  } else if (cache_mode == sgl_xpu::NvFp4WideCache::Slm) {
    switch (tile) {
      case MTile::M8N128:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n128_policy<true>);
        break;
      case MTile::M8N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n256_policy<true>);
        break;
      case MTile::M32N256:
        // The M32 SLM specialization stalls on Xe3P.  The M64 policy is
        // bit-exact for partial-M tiles and uses the proven cache path.
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m64n256_policy<true, sgl_xpu::NvFp4WideCache::Slm>);
        break;
      case MTile::M64N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m64n256_policy<true, sgl_xpu::NvFp4WideCache::Slm>);
        break;
      case MTile::M256N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m256n256_policy<true, sgl_xpu::NvFp4WideCache::Slm>);
        break;
    }
  } else if (nvfp4_woq_quantize_in_k_loop_enabled()) {
    switch (tile) {
      case MTile::M8N128:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n128_policy<true>);
        break;
      case MTile::M8N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n256_policy<true>);
        break;
      case MTile::M32N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m32n256_policy<true>);
        break;
      case MTile::M64N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m64n256_policy<true>);
        break;
      case MTile::M256N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m256n256_policy<true>);
        break;
    }
  } else {
    switch (tile) {
      case MTile::M8N128:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n128_policy<false>);
        break;
      case MTile::M8N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m8n256_policy<false>);
        break;
      case MTile::M32N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m32n256_policy<false>);
        break;
      case MTile::M64N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m64n256_policy<false>);
        break;
      case MTile::M256N256:
        RUN_MXFP8_POLICY(policies::moe_nvfp4_mxfp8_m256n256_policy<false>);
        break;
    }
  }

#undef RUN_MXFP8_POLICY
}

void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq_bf16: A must be on XPU");
  const auto device = a.device();
  TORCH_CHECK(
      b.device() == device && sfmx_b.device() == device && c_b.device() == device && output.device() == device &&
          workspace.device() == device,
      "nvfp4_woq_bf16: all tensors must be on the same XPU device");
  check_group_metadata(problem_sizes, expert_offsets, device);

  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: A must be bfloat16");
  TORCH_CHECK(
      b.scalar_type() == torch::kUInt8 && sfmx_b.scalar_type() == torch::kUInt8,
      "nvfp4_woq_bf16: B and sfmx_b must be uint8");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq_bf16: output must be float32");
  TORCH_CHECK(
      a.dim() == 2 && output.dim() == 2 && b.dim() == 3 && sfmx_b.dim() == 3 && c_b.dim() == 3,
      "nvfp4_woq_bf16: invalid tensor ranks");
  TORCH_CHECK(
      a.is_contiguous() && b.is_contiguous() && sfmx_b.is_contiguous() && c_b.is_contiguous() && output.is_contiguous(),
      "nvfp4_woq_bf16: all data tensors must be contiguous");

  const int64_t groups = problem_sizes.size(0);
  const int64_t total_m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = output.size(1);
  TORCH_CHECK(groups > 0 && total_m > 0, "nvfp4_woq_bf16: E and total M must be positive");
  TORCH_CHECK(k % 32 == 0, "nvfp4_woq_bf16: K must be a multiple of 32");
  TORCH_CHECK(n % 4 == 0, "nvfp4_woq_bf16: N must be a multiple of 4");
  TORCH_CHECK(output.size(0) == total_m, "nvfp4_woq_bf16: A and output row counts must match");
  TORCH_CHECK(b.size(0) == groups && b.size(1) == n && b.size(2) == k / 2, "nvfp4_woq_bf16: B must be (E, N, K/2)");
  TORCH_CHECK(
      sfmx_b.size(0) == groups && sfmx_b.size(1) == k / kMxBlockSize && sfmx_b.size(2) == n,
      "nvfp4_woq_bf16: sfmx_b must be (E, K/32, N)");
  TORCH_CHECK(
      c_b.size(0) == groups && c_b.size(1) == n && c_b.size(2) == k / kNvFp4BlockSize,
      "nvfp4_woq_bf16: c_b must be (E, N, K/16)");

  auto rows_per_expert = problem_sizes.select(1, 0).contiguous();
  auto& queue = at::xpu::getCurrentXPUStream(device.index()).queue();
  const auto hw_info = hardware_info(device);
  const auto tile = select_m_tile(total_m, groups, n, hw_info.sm_count);

#define RUN_BF16_POLICY(POLICY) \
  run_bf16_policy<POLICY>(queue, hw_info, rows_per_expert, a, b, sfmx_b, c_b, output, n, k, groups)

  switch (tile) {
    case MTile::M8N128:
      RUN_BF16_POLICY(policies::moe_nvfp4_bf16_m8n128_policy);
      break;
    case MTile::M8N256:
      RUN_BF16_POLICY(policies::moe_nvfp4_bf16_m8n256_policy);
      break;
    case MTile::M32N256:
      RUN_BF16_POLICY(policies::moe_nvfp4_bf16_m32n256_policy);
      break;
    case MTile::M64N256:
      RUN_BF16_POLICY(policies::moe_nvfp4_bf16_m64n256_policy);
      break;
    case MTile::M256N256:
      RUN_BF16_POLICY(policies::moe_nvfp4_bf16_m256n256_policy);
      break;
  }

#undef RUN_BF16_POLICY
}

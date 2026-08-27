/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <ATen/ATen.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int64_t kDefaultBlock = 256;
constexpr int64_t kRelProjVec = 8;

inline int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t round_up_i64(int64_t x, int64_t multiple) {
  return ceil_div_i64(x, multiple) * multiple;
}

inline float bf16_raw_to_float(uint16_t raw) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(raw) << 16);
}

inline uint16_t bf16_float_to_raw(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_to_float(bf16_t value) {
  return bf16_raw_to_float(sycl::bit_cast<uint16_t>(value));
}

inline bf16_t float_to_bf16(float value) {
  return sycl::bit_cast<bf16_t>(bf16_float_to_raw(value));
}

struct RelProjParams {
  const bf16_t* r = nullptr;
  const bf16_t* proj = nullptr;
  const float* tau = nullptr;
  bf16_t* out = nullptr;
  int64_t t = 0;
  int64_t h = 0;
  int64_t d = 0;
  int64_t e = 0;
  int64_t r_stride_t = 0;
};

template <bool HasTau>
class InklingRelProjBf16D16EsimdKernel {
 public:
  RelProjParams params;
  int64_t chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int64_t linear = static_cast<int64_t>(item.get_linear_id());
    int64_t total = params.t * params.h * chunks_per_row;
    if (linear >= total) {
      return;
    }

    int64_t chunk = linear % chunks_per_row;
    int64_t th = linear / chunks_per_row;
    int64_t ti = th / params.h;
    int64_t hi = th - ti * params.h;
    int64_t e0 = chunk * 16;

    float scale = 1.0f;
    if constexpr (HasTau) {
      scale = params.tau[ti];
    }

    sycl::ext::intel::esimd::simd<float, 8> acc_lo(0.0f);
    sycl::ext::intel::esimd::simd<float, 8> acc_hi(0.0f);
    const bf16_t* r_row = params.r + ti * params.r_stride_t + hi * params.d;

#pragma unroll
    for (int d = 0; d < 16; ++d) {
      float r_value = bf16_to_float(r_row[d]);
      if constexpr (HasTau) {
        r_value = bf16_to_float(float_to_bf16(r_value * scale));
      }

      auto raw = sycl::ext::intel::esimd::block_load<uint32_t, 8>(
          reinterpret_cast<const uint32_t*>(params.proj + static_cast<int64_t>(d) * params.e) + e0 / 2);
      auto lo_bits = (raw & 0x0000ffffu) << 16;
      auto hi_bits = raw & 0xffff0000u;
      acc_lo += lo_bits.template bit_cast_view<float>() * r_value;
      acc_hi += hi_bits.template bit_cast_view<float>() * r_value;
    }

    auto lo_bits = acc_lo.template bit_cast_view<uint32_t>();
    auto hi_bits = acc_hi.template bit_cast_view<uint32_t>();
    auto lo_round = ((lo_bits >> 16) & 1u) + 0x7fffu;
    auto hi_round = ((hi_bits >> 16) & 1u) + 0x7fffu;
    auto packed = ((lo_bits + lo_round) >> 16) | (((hi_bits + hi_round) >> 16) << 16);

    bf16_t* out_row = params.out + (ti * params.h + hi) * params.e;
    sycl::ext::intel::esimd::block_store<uint32_t, 8>(reinterpret_cast<uint32_t*>(out_row) + e0 / 2, packed);
  }
};

template <bool HasTau>
void launch_rel_proj_bf16_d16_esimd(sycl::queue& queue, const RelProjParams& params) {
  int64_t chunks_per_row = params.e / 16;
  int64_t total = params.t * params.h * chunks_per_row;
  InklingRelProjBf16D16EsimdKernel<HasTau> kernel{params, chunks_per_row};
  queue.parallel_for<InklingRelProjBf16D16EsimdKernel<HasTau>>(sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

template <bool HasTau, int64_t Vec>
class InklingRelProjKernel {
 public:
  RelProjParams params;
  int64_t total;
  int64_t e_vecs;

  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    int64_t e_vec = idx % e_vecs;
    int64_t th = idx / e_vecs;
    int64_t ti = th / params.h;
    int64_t hi = th - ti * params.h;
    int64_t e0 = e_vec * Vec;

    float scale = 1.0f;
    if constexpr (HasTau) {
      scale = params.tau[ti];
    }

    float acc[Vec];
#pragma unroll
    for (int64_t i = 0; i < Vec; ++i) {
      acc[i] = 0.0f;
    }

    const bf16_t* r_row = params.r + ti * params.r_stride_t + hi * params.d;
    for (int64_t d = 0; d < params.d; ++d) {
      float r_value = bf16_to_float(r_row[d]);
      if constexpr (HasTau) {
        r_value = bf16_to_float(float_to_bf16(r_value * scale));
      }

      const bf16_t* proj_row = params.proj + d * params.e;
#pragma unroll
      for (int64_t i = 0; i < Vec; ++i) {
        int64_t e_col = e0 + i;
        if (e_col < params.e) {
          acc[i] += r_value * bf16_to_float(proj_row[e_col]);
        }
      }
    }

    bf16_t* out_row = params.out + (ti * params.h + hi) * params.e;
#pragma unroll
    for (int64_t i = 0; i < Vec; ++i) {
      int64_t e_col = e0 + i;
      if (e_col < params.e) {
        out_row[e_col] = float_to_bf16(acc[i]);
      }
    }
  }
};

template <bool HasTau, int64_t Vec = kRelProjVec>
void launch_rel_proj_kernel(sycl::queue& queue, const RelProjParams& params) {
  int64_t e_vecs = ceil_div_i64(params.e, Vec);
  int64_t total = params.t * params.h * e_vecs;
  if (total == 0) {
    return;
  }

  if (params.t <= 4 && params.d == 16 && params.e % 16 == 0) {
    launch_rel_proj_bf16_d16_esimd<HasTau>(queue, params);
    return;
  }

  int64_t global = round_up_i64(total, kDefaultBlock);
  InklingRelProjKernel<HasTau, Vec> kernel{params, total, e_vecs};
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)), sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
        kernel);
  });
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor inkling_rel_proj_small_t(
    const at::Tensor& r, const at::Tensor& proj, const std::optional<at::Tensor>& tau, const at::Tensor& out) {
  CHECK_DEVICE(r);
  CHECK_DEVICE(proj);
  CHECK_DEVICE(out);
  TORCH_CHECK(r.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: r must be bfloat16");
  TORCH_CHECK(proj.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: proj must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: out must be bfloat16");
  TORCH_CHECK(r.dim() == 3, "inkling_rel_proj_small_t: r must have shape [t, h, d]");
  TORCH_CHECK(proj.dim() == 2, "inkling_rel_proj_small_t: proj must have shape [d, e]");
  TORCH_CHECK(out.dim() == 3, "inkling_rel_proj_small_t: out must have shape [t, h, e]");
  TORCH_CHECK(r.size(2) == proj.size(0), "inkling_rel_proj_small_t: r.size(2) must equal proj.size(0)");
  TORCH_CHECK(out.size(0) == r.size(0), "inkling_rel_proj_small_t: out.size(0) must equal r.size(0)");
  TORCH_CHECK(out.size(1) == r.size(1), "inkling_rel_proj_small_t: out.size(1) must equal r.size(1)");
  TORCH_CHECK(out.size(2) == proj.size(1), "inkling_rel_proj_small_t: out.size(2) must equal proj.size(1)");
  TORCH_CHECK(r.stride(2) == 1, "inkling_rel_proj_small_t: r must be contiguous on the d dimension");
  TORCH_CHECK(r.stride(1) == r.size(2), "inkling_rel_proj_small_t: r must have contiguous [h, d] rows");
  TORCH_CHECK(proj.is_contiguous(), "inkling_rel_proj_small_t: proj must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "inkling_rel_proj_small_t: out must be contiguous");

  const float* tau_ptr = nullptr;
  bool has_tau = tau.has_value();
  if (has_tau) {
    const at::Tensor& tau_tensor = tau.value();
    CHECK_DEVICE(tau_tensor);
    TORCH_CHECK(tau_tensor.scalar_type() == at::ScalarType::Float, "inkling_rel_proj_small_t: tau must be float32");
    TORCH_CHECK(tau_tensor.dim() == 1, "inkling_rel_proj_small_t: tau must be 1D");
    TORCH_CHECK(tau_tensor.numel() == r.size(0), "inkling_rel_proj_small_t: tau must have one entry per token");
    TORCH_CHECK(tau_tensor.is_contiguous(), "inkling_rel_proj_small_t: tau must be contiguous");
    tau_ptr = tau_tensor.data_ptr<float>();
  }

  RelProjParams params{};
  params.r = reinterpret_cast<const bf16_t*>(r.data_ptr<at::BFloat16>());
  params.proj = reinterpret_cast<const bf16_t*>(proj.data_ptr<at::BFloat16>());
  params.tau = tau_ptr;
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.t = r.size(0);
  params.h = r.size(1);
  params.d = r.size(2);
  params.e = proj.size(1);
  params.r_stride_t = r.stride(0);

  auto queue = dpcppGetCurrentQueue();
  if (has_tau) {
    launch_rel_proj_kernel<true>(queue, params);
  } else {
    launch_rel_proj_kernel<false>(queue, params);
  }
  return out;
}

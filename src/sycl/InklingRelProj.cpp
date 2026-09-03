/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <ATen/ATen.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Numerics.h"
#include "sgl_kernel_export.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int64_t kDefaultBlock = 256;
constexpr int64_t kRelProjVec = 8;

// Production Inkling shapes: r is [t, h, 16]. Global attention uses
// proj=[16, 1024], while local attention uses proj=[16, 512].
constexpr int64_t kRelProjD = 16;
constexpr int64_t kRelProjGlobalE = 1024;
constexpr int64_t kRelProjLocalE = 512;

inline float bf16_raw_to_float(uint16_t raw) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(raw) << 16);
}

inline uint16_t bf16_float_to_raw(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_round_trip(float value) {
#if defined(__SYCL_DEVICE_ONLY__)
  return static_cast<float>(static_cast<bf16_t>(value));
#else
  return bf16_raw_to_float(bf16_float_to_raw(value));
#endif
}

struct RelProjParams {
  const bf16_t* r = nullptr;
  const bf16_t* proj = nullptr;
  const float* tau = nullptr;
  bf16_t* out = nullptr;
  int64_t t = 0;
  int64_t h = 0;
  int64_t e = 0;
  int64_t r_stride_t = 0;
};

// A runtime integer division costs tens of instructions on Xe and is a
// substantial part of the tiny decode launch. The two divisors here are known
// at launch time, so use a host-computed multiply-shift when it is exact.
#if !defined(SGL_INKLING_RELPROJ_FAST_DIV)
#define SGL_INKLING_RELPROJ_FAST_DIV 1
#endif
#if !defined(SGL_INKLING_RELPROJ_FAST_DIV_VERIFY)
#define SGL_INKLING_RELPROJ_FAST_DIV_VERIFY 0
#endif

struct RelProjFastDiv {
  uint32_t magic = 0;
  int shift = -1;  // negative falls back to a real division
};

inline RelProjFastDiv make_rel_proj_fast_div_magic(int divisor, int max_value) {
  RelProjFastDiv fd;
  if (divisor <= 0 || max_value < 0) {
    return fd;
  }
  if (max_value < divisor) {
    fd.magic = 0;
    fd.shift = 0;
    return fd;
  }
  int l = 0;
  while ((1 << l) < divisor) {
    ++l;
  }
  if ((1 << l) == divisor) {
    fd.magic = 1;
    fd.shift = l;
    return fd;
  }

  uint64_t need = static_cast<uint64_t>(max_value) * static_cast<uint64_t>(divisor) + 1ull;
  int shift = 0;
  while ((1ull << shift) < need) {
    ++shift;
  }
  uint64_t magic = (1ull << shift) / static_cast<uint64_t>(divisor) + 1ull;
  if (magic > 0xffffffffull || magic * static_cast<uint64_t>(max_value) > 0xffffffffull) {
    return fd;
  }
  fd.magic = static_cast<uint32_t>(magic);
  fd.shift = shift;
  return fd;
}

inline RelProjFastDiv make_rel_proj_fast_div(int divisor, int max_value) {
#if SGL_INKLING_RELPROJ_FAST_DIV
  RelProjFastDiv fd = make_rel_proj_fast_div_magic(divisor, max_value);
#if SGL_INKLING_RELPROJ_FAST_DIV_VERIFY
  if (fd.shift >= 0) {
    for (int value = 0; value <= max_value; ++value) {
      int got = static_cast<int>((static_cast<uint32_t>(value) * fd.magic) >> fd.shift);
      if (got != value / divisor) {
        std::cerr << "inkling rel_proj fast div is wrong: " << value << " / " << divisor << " gave " << got
                  << ", expected " << (value / divisor) << "\n";
        std::abort();
      }
    }
  }
#endif
  return fd;
#else
  (void)divisor;
  (void)max_value;
  return RelProjFastDiv{};
#endif
}

inline int rel_proj_fast_div(int value, RelProjFastDiv fd, int divisor) {
#if SGL_INKLING_RELPROJ_FAST_DIV
  if (fd.shift < 0) {
    return value / divisor;
  }
  return static_cast<int>((static_cast<uint32_t>(value) * fd.magic) >> fd.shift);
#else
  (void)fd;
  return value / divisor;
#endif
}

// out[t, h, :] = bf16(tau[t] * r[t, h, :]) @ proj
//
// One work-item owns Vec output columns, keeps that projection slice in
// registers, and applies it to MTile rows. This avoids re-reading proj for
// every output row while retaining enough parallelism for decode-sized shapes.
template <int MTile, int Vec>
class InklingRelProjKernel {
 public:
  static_assert(Vec % 2 == 0, "Vec must be even so proj/out move as 32-bit pairs");

  RelProjParams params;
  int total;
  int col_slices;
  RelProjFastDiv col_div;
  RelProjFastDiv h_div;

  void operator()(sycl::nd_item<1> item) const {
    int idx = static_cast<int>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    int m_tile = rel_proj_fast_div(idx, col_div, col_slices);
    int col_slice = idx - m_tile * col_slices;
    int e0 = col_slice * Vec;

    float proj_tile[kRelProjD][Vec];
#pragma unroll
    for (int d = 0; d < kRelProjD; ++d) {
      const uint32_t* proj_row = reinterpret_cast<const uint32_t*>(params.proj + static_cast<int64_t>(d) * params.e);
#pragma unroll
      for (int i = 0; i < Vec / 2; ++i) {
        uint32_t pair = proj_row[e0 / 2 + i];
        proj_tile[d][2 * i] = bf16_raw_to_float(static_cast<uint16_t>(pair & 0xffffu));
        proj_tile[d][2 * i + 1] = bf16_raw_to_float(static_cast<uint16_t>(pair >> 16));
      }
    }

    int m = m_tile * MTile;
    int rows = static_cast<int>(params.t * params.h);
    int ti = rel_proj_fast_div(m, h_div, static_cast<int>(params.h));
    int hi = m - ti * params.h;

#pragma unroll
    for (int mm = 0; mm < MTile; ++mm) {
      if (m >= rows) {
        return;
      }

      float scale = params.tau[ti];
      const bf16_t* r_row = params.r + static_cast<int64_t>(ti) * params.r_stride_t + hi * kRelProjD;

      float acc[Vec];
#pragma unroll
      for (int i = 0; i < Vec; ++i) {
        acc[i] = 0.0f;
      }

      const uint32_t* r_pairs = reinterpret_cast<const uint32_t*>(r_row);
#pragma unroll
      for (int d = 0; d < kRelProjD; d += 2) {
        uint32_t pair = r_pairs[d / 2];
        float r_lo = bf16_raw_to_float(static_cast<uint16_t>(pair & 0xffffu));
        float r_hi = bf16_raw_to_float(static_cast<uint16_t>(pair >> 16));
        r_lo = bf16_round_trip(r_lo * scale);
        r_hi = bf16_round_trip(r_hi * scale);
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
          acc[i] += r_lo * proj_tile[d][i];
          acc[i] += r_hi * proj_tile[d + 1][i];
        }
      }

      uint32_t* out_row = reinterpret_cast<uint32_t*>(params.out + static_cast<int64_t>(m) * params.e);
#pragma unroll
      for (int i = 0; i < Vec / 2; ++i) {
        out_row[e0 / 2 + i] = static_cast<uint32_t>(bf16_float_to_raw(acc[2 * i])) |
                              (static_cast<uint32_t>(bf16_float_to_raw(acc[2 * i + 1])) << 16);
      }

      ++m;
      if (++hi == params.h) {
        hi = 0;
        ++ti;
      }
    }
  }
};

struct RelProjLaunchPlan {
  int vec = kRelProjVec;
  int local = kDefaultBlock;
  int mtile = 1;
};

inline RelProjLaunchPlan rel_proj_launch_plan(int rows) {
  if (rows <= 12) {
    return {2, 16, 1};
  }
  if (rows <= 40) {
    return {2, 64, 1};
  }
  if (rows <= 64) {
    return {8, 16, 1};
  }
  if (rows <= 256) {
    constexpr int kMinRowTiles = 40;
    int mtile = 1;
    while (mtile < 16 && rows >= kMinRowTiles * mtile * 2) {
      mtile *= 2;
    }
    return {4, 64, mtile};
  }
  return {8, 16, 1};
}

template <int MTile, int Vec>
void submit_rel_proj_kernel(sycl::queue& queue, const RelProjParams& params, int local_size) {
  int col_slices = static_cast<int>(params.e / Vec);
  int total = static_cast<int>(CeilDiv(params.t * params.h, static_cast<int64_t>(MTile)) * col_slices);
  int local = std::max(1, std::min(local_size, total));
  int global = RoundUp(total, local);
  InklingRelProjKernel<MTile, Vec> kernel{
      params,
      total,
      col_slices,
      make_rel_proj_fast_div(col_slices, total),
      make_rel_proj_fast_div(static_cast<int>(params.h), static_cast<int>(params.t * params.h))};
  sycl_kernel_submit(global, local, queue, kernel);
}

void launch_rel_proj_kernel(sycl::queue& queue, const RelProjParams& params) {
  RelProjLaunchPlan plan = rel_proj_launch_plan(static_cast<int>(params.t * params.h));
  if (plan.vec == 2) {
    submit_rel_proj_kernel<1, 2>(queue, params, plan.local);
    return;
  }
  if (plan.vec == 4) {
    switch (plan.mtile) {
      case 1:
        submit_rel_proj_kernel<1, 4>(queue, params, plan.local);
        return;
      case 2:
        submit_rel_proj_kernel<2, 4>(queue, params, plan.local);
        return;
      case 4:
        submit_rel_proj_kernel<4, 4>(queue, params, plan.local);
        return;
      case 8:
        submit_rel_proj_kernel<8, 4>(queue, params, plan.local);
        return;
      default:
        submit_rel_proj_kernel<16, 4>(queue, params, plan.local);
        return;
    }
  }
  submit_rel_proj_kernel<1, 8>(queue, params, plan.local);
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor
inkling_rel_proj_small_t(const at::Tensor& r, const at::Tensor& proj, const at::Tensor& tau, const at::Tensor& out) {
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
  TORCH_CHECK(r.size(2) == kRelProjD, "inkling_rel_proj_small_t: only production d_rel=16 is supported");
  TORCH_CHECK(
      proj.size(1) == kRelProjGlobalE || proj.size(1) == kRelProjLocalE,
      "inkling_rel_proj_small_t: only production rel_extent=1024 or local_extent=512 is supported");
  TORCH_CHECK(
      r.size(0) == 1 || r.stride(0) > r.size(1) * r.size(2),
      "inkling_rel_proj_small_t: r must be the strided trailing view of the packed qkvr output");
  TORCH_CHECK(proj.is_contiguous(), "inkling_rel_proj_small_t: proj must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "inkling_rel_proj_small_t: out must be contiguous");

  CHECK_DEVICE(tau);
  TORCH_CHECK(tau.scalar_type() == at::ScalarType::Float, "inkling_rel_proj_small_t: tau must be float32");
  TORCH_CHECK(tau.dim() == 1, "inkling_rel_proj_small_t: tau must be 1D");
  TORCH_CHECK(tau.numel() == r.size(0), "inkling_rel_proj_small_t: tau must have one entry per token");
  TORCH_CHECK(tau.is_contiguous(), "inkling_rel_proj_small_t: tau must be contiguous");

  RelProjParams params{};
  params.r = reinterpret_cast<const bf16_t*>(r.data_ptr<at::BFloat16>());
  params.proj = reinterpret_cast<const bf16_t*>(proj.data_ptr<at::BFloat16>());
  params.tau = tau.data_ptr<float>();
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.t = r.size(0);
  params.h = r.size(1);
  params.e = proj.size(1);
  params.r_stride_t = r.stride(0);

  // The kernel moves proj and out as bf16 pairs.
  TORCH_CHECK(params.r_stride_t % 2 == 0, "inkling_rel_proj_small_t: r token stride must be 4-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(params.proj) % sizeof(uint32_t) == 0,
      "inkling_rel_proj_small_t: proj must be 4-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(params.out) % sizeof(uint32_t) == 0,
      "inkling_rel_proj_small_t: out must be 4-byte aligned");

  auto queue = dpcppGetCurrentQueue();
  launch_rel_proj_kernel(queue, params);
  return out;
}

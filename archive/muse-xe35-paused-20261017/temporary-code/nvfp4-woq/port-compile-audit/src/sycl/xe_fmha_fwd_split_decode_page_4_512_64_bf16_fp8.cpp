/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Auto-generated from xe_fmha_fwd_split_decode_fp8_kernel.cpp.in
// Template parameters: QG_SZ=4, HEAD_DIM=512, PAGE_SIZE=64, ELEM_TAG=bf16
//
// FP8 KV-cache split-KV decode path, split out of
// xe_fmha_fwd_split_decode_kernel.cpp.in so the e4m3/e5m2 kernel instantiations
// compile in their own translation unit and keep the peak compiler memory of
// the (bf16/fp16) split decode TU low.
#define SYCL_INTEL_TARGET 20

#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_runner.hpp"

namespace decode {

template <>
__attribute__((visibility("default"))) void FmhaSplitDecodeFp8Runner<4, 512, 64, cutlass::bfloat16_t>::operator()(const Arguments& params) const {
  TORCH_CHECK(params.rel_bias_ptr == nullptr, "relative attention does not support fp8 KV cache");
  using TileShapeQK = cute::Shape<cute::Int<4>, cute::Int<64>, cute::_64>;
  using TileShapePV = cute::Shape<cute::Int<4>, cute::_32, cute::Int<64>>;
  static constexpr int kOutputTile = 512 == 512 ? 256 : ((512 + 31) / 32) * 32;
  using TileShapeOutput = cute::Shape<cute::Int<4>, cute::Int<kOutputTile>>;
  // Cap KV subgroups so the epilogue cross-subgroup reduction buffer fits in shared
  // local memory (see kv_subgroups_for_slm); the full PAGE_SIZE/16 split overflows SLM
  // for large head dims / GQA groups / page sizes and hangs the device.
  static constexpr int kKvSubgroups =
      kv_subgroups_for_slm(64 / 16, 4, kOutputTile);
  using SubgroupLayoutQK = cute::Layout<cute::Shape<cute::_1, cute::Int<kKvSubgroups>, cute::_1>>;

  // FP8 KV cache: Q is bf16 or fp16 while K/V cache are fp8 (e4m3 or e5m2); the
  // split-KV mainloop dequantizes K/V with the per-tensor k/v descale. Sink
  // logits are not supported on the fp8 path.
  TORCH_CHECK(params.is_e4m3 || params.is_e5m2, "FmhaSplitDecodeFp8Runner requires an fp8 KV cache");
  TORCH_CHECK(!params.use_sink, "fp8 KV cache decode does not support sink logits");
  // No /*LSE=*/true instantiation is compiled for the fp8 KV path (it would
  // double this already heavy translation unit), so reject the request instead
  // of leaving the caller's softmax_lse buffer unwritten.
  TORCH_CHECK(
      !params.return_softmax_lse, "return_softmax_lse is not supported with an fp8 KV cache (split decode)");
  AT_DISPATCH_BOOL_NO_RETURN(params.is_local, LocalMask, {
    using QElement = cutlass::bfloat16_t;
    dispatch_softcap<512, true>(params.softcap, [&](auto softcap_tag) {
      constexpr bool Softcap = decltype(softcap_tag)::value;
    if (params.is_e4m3) {
      SplitDecodeConfig<
          false, LocalMask, false, Softcap, /*LSE=*/false, TileShapeQK, TileShapePV, TileShapeOutput, SubgroupLayoutQK, void, 1,
          QElement, cutlass::float_e4m3_t, cutlass::float_e4m3_t, QElement>::run(params);
    } else {
      SplitDecodeConfig<
          false, LocalMask, false, Softcap, /*LSE=*/false, TileShapeQK, TileShapePV, TileShapeOutput, SubgroupLayoutQK, void, 1,
          QElement, cutlass::float_e5m2_t, cutlass::float_e5m2_t, QElement>::run(params);
    }
    });
  });
}

template struct FmhaSplitDecodeFp8Runner<4, 512, 64, cutlass::bfloat16_t>;

}  // namespace decode

#ifdef SGL_FMHA_JIT_ENTRY
extern "C" __attribute__((visibility("default"))) void sgl_fmha_entry(const void* params) {
  decode::FmhaSplitDecodeFp8Runner<4, 512, 64, cutlass::bfloat16_t>{}(
      *static_cast<const decode::Arguments*>(params));
}
#endif

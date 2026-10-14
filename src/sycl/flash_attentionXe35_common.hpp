#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <limits>
#include <optional>
#include <type_traits>

#include "comm/common.h"
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/sycl_event_manager.hpp>
#endif
#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_epilogue.hpp"
#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_mainloop.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_fmha_fwd_decode_splitkv_kernel.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_fmha_fwd_kernel.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_reduce_split_k.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_tile_scheduler.hpp"

#define SYCL_INTEL_TARGET 35

using namespace cute;

namespace xe35_fmha {

namespace detail {
struct SplitKVCache {
  at::Tensor temp_out;
  at::Tensor exp_sums;
  at::Tensor max_logits;

  void
  ensure(int64_t out_elts, int64_t lse_elts, const at::TensorOptions& out_opts, const at::TensorOptions& lse_opts) {
    if (!temp_out.defined() || temp_out.numel() < out_elts || temp_out.dtype() != out_opts.dtype() ||
        temp_out.device() != out_opts.device()) {
      temp_out = at::empty({out_elts}, out_opts);
    }
    if (!exp_sums.defined() || exp_sums.numel() < lse_elts || exp_sums.dtype() != lse_opts.dtype() ||
        exp_sums.device() != lse_opts.device()) {
      exp_sums = at::empty({lse_elts}, lse_opts);
    }
    if (!max_logits.defined() || max_logits.numel() < lse_elts || max_logits.dtype() != lse_opts.dtype() ||
        max_logits.device() != lse_opts.device()) {
      max_logits = at::empty({lse_elts}, lse_opts);
    }
  }
};

inline SplitKVCache& get_split_kv_cache() {
  thread_local SplitKVCache cache;
  return cache;
}
}  // namespace detail
// FMHA must run on the caller's torch stream, not on a private queue.
//
// A private queue shares the context and device with torch's stream but nothing
// orders the two, so (a) attention neither waits for the producers of q/k/v nor
// is waited on by its consumers, and (b) the torch caching allocator's
// stream-ordered reuse guarantee is void: `out`, `softmax_lse` and the decode
// workspace are freed when the op returns while the kernel is still writing
// them, and the next torch-stream allocation lands on those blocks. That was
// observed as a grouped-MoE GEMM reading bf16 attention data in place of its
// (M,N,K) table and pointer array, producing block-2D descriptors with absurd
// extents -> prefetch faults -> engine reset -> UR_RESULT_ERROR_DEVICE_LOST.
inline sycl::queue& torch_stream_queue(const at::Tensor& t) {
  return at::xpu::getCurrentXPUStream(t.device().index()).queue();
}

// Persistent split-KV workspace for the decode kernel: one int counter per
// (batch, kv head) followed by the per-split partial blocks.  Workspaces are
// private to each torch stream: calls on one stream are ordered, while calls
// on different streams may execute concurrently.  The kernel resets counters
// after the last split, so a stream's buffer only needs zeroing when allocated.
// Growing it is safe through the torch stream-ordered caching allocator.
struct DecodeSplitKvWorkspace {
  at::Tensor buffer;
  size_t counter_bytes = 0;
  size_t partial_bytes = 0;
};

inline void decode_splitkv_workspace(
    const at::Tensor& like, int slots, int num_splits, size_t bytes_per_split, int** counters, void** partials) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unordered_map<int64_t, DecodeSplitKvWorkspace>> per_stream;
  std::lock_guard<std::mutex> guard(mutex);
  const auto stream_id = static_cast<int64_t>(at::xpu::getCurrentXPUStream(like.device().index()).id());
  auto& ws = per_stream[like.device().index()][stream_id];
  const size_t need_counter_bytes = ((size_t(slots) * sizeof(int) + 255) / 256) * 256;
  const size_t need_partial_bytes = size_t(slots) * num_splits * bytes_per_split;
  if (!ws.buffer.defined() || ws.counter_bytes < need_counter_bytes || ws.partial_bytes < need_partial_bytes) {
    ws.counter_bytes = std::max(ws.counter_bytes, need_counter_bytes);
    ws.partial_bytes = std::max(ws.partial_bytes, need_partial_bytes);
    ws.buffer = at::zeros({static_cast<long>(ws.counter_bytes + ws.partial_bytes)}, like.options().dtype(at::kByte));
  }
  auto* base = static_cast<char*>(ws.buffer.data_ptr());
  *counters = reinterpret_cast<int*>(base);
  *partials = base + ws.counter_bytes;
}

// Number of KV splits per (batch, kv head) for decode.
//
// Measured on CRI (Muse Spark L3 shape, 4097 cached tokens, BF16): with one
// split the kernel streams K/V at ~1.0 TB/s, the device DRAM ceiling, once
// there are >= 64 workgroups (batch >= 4).  Splitting a slot further is
// harmful there: the split workgroups of one sequence read regions ~1-4 MB
// apart at the same time and DRAM efficiency drops roughly in proportion to
// the split count (B=8: 131 us at 1 split, 300 us at 8 splits).  Splits only
// pay off when the grid is too small to cover the memory system (B=1: 53 us
// at 1 split, 27 us at 4 splits), so target ~64 workgroups in total.
inline int choose_decode_kv_splits(int slots, int max_tiles) {
  constexpr int target_wgs = 64;
  constexpr int max_splits = 8;
  constexpr int min_tiles_per_split = 4;
  int splits = (target_wgs + std::max(1, slots) - 1) / std::max(1, slots);
  splits = std::min(splits, std::max(1, max_splits));
  splits = std::min(splits, std::max(1, max_tiles / std::max(1, min_tiles_per_split)));
  return std::max(1, splits);
}

template <class Element, class OutputElement = Element>
struct Xe3FmhaConfig {
  using ElementInput = Element;
  using ElementOutput = OutputElement;
  using ElementScale = float;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  template <
      bool Causal,
      bool LocalMask,
      bool Sink,
      int PipelineStages,
      bool GqaFusion,
      bool Paged,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut,
      class SubgroupLayoutQK,
      int FixedWindowLeft = -2,
      int FixedWindowRight = -2,
      int FixedPageSize = -1,
      XeStoreCachePolicy StoreCachePolicyO = XeStoreCachePolicy::kWT_WB_WB>
  static void
  run(const at::Tensor& q_dense,
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
      int head_dim,
      int page_size,
      float q_descale,
      float k_descale,
      float v_descale,
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink,
      int num_kv_splits = 1) {
    static constexpr bool BlockScale = false;
    static constexpr bool F8kvF16mma = false;
    static constexpr bool PerTensorScale = sizeof_bits_v<ElementInput> == 8;
    // Paged KV is always cached; non-paged runs as plain attention (CachedKV=false)
    // with the KV placed in the "new" slot, which is the upstream-tested path.
    static constexpr bool CachedKV = Paged;
    // Paged (PagedKV=true) uses page_table indirection; non-paged (Paged=false)
    // reads contiguous ragged KV addressed by the cumulative cu_seqlens_k_cache.
    static constexpr bool PagedKV = Paged;

    constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q dimension");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        LocalMask,
        BlockScale,
        F8kvF16mma,
        PerTensorScale,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void,
        FixedWindowLeft,
        FixedWindowRight,
        FixedPageSize>;

    using CollectiveEpilogue = cutlass::fmha::collective::
        FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink, StoreCachePolicyO>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    // Single-stage paged local attention has no new-K/V path.
    static constexpr bool CacheOnly = Paged && LocalMask && PipelineStages == 1;
    using Scheduler = cutlass::fmha::kernel::
        XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion, CacheOnly, GqaFusion ? 2 : 0>;
    using FMHAKernel =
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;
    using SplitScheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, true>;
    using SplitKernel = cutlass::fmha::kernel::
        XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, SplitScheduler, true>;
    using ReduceKernel = cutlass::reduction::kernel::
        ReduceSplitK<ProblemShape, cutlass::fmha::kernel::XeReduceSplitKTileScheduler, SplitKernel>;

    const int q_seq_stride = static_cast<int>(q_dense.stride(0));
    const int q_head_stride = static_cast<int>(q_dense.stride(1));
    const int kv_seq_stride = static_cast<int>(k_pool.stride(0));
    const int kv_head_stride = static_cast<int>(k_pool.stride(1));
    const int o_seq_stride = static_cast<int>(out_dense.stride(0));
    const int o_head_stride = static_cast<int>(out_dense.stride(1));
    typename FMHAKernel::StrideQ stride_q{q_seq_stride, {}, q_head_stride, 0};
    typename FMHAKernel::StrideK stride_k{kv_seq_stride, {}, kv_head_stride, 0};
    typename FMHAKernel::StrideV stride_v{{}, kv_seq_stride, kv_head_stride, 0};
    typename FMHAKernel::StrideO stride_o{o_seq_stride, {}, o_head_stride, 0};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // page_table is only valid in the paged case; keep it null otherwise so the
    // non-paged mainloop never dereferences an absent table.
    const int* page_table_ptr = nullptr;
    if constexpr (PagedKV) {
      page_table_ptr = static_cast<const int*>(page_table.data_ptr());
    }

    // Plain attention (CachedKV=false) carries the KV in the "new" slot; paged
    // carries it in the "cache" slot. The unused slot stays empty.
    int* const cu_kv = cu_seqlens_k_cache.data_ptr<int>();
    const int kv_total = static_cast<int>(k_pool.size(0));
    const int new_max = Paged ? 0 : max_seqlen_kv_cache;
    const int new_total = Paged ? 0 : kv_total;
    int* const new_cu = Paged ? nullptr : cu_kv;
    const int cache_max = Paged ? max_seqlen_kv_cache : 0;
    const int cache_total = Paged ? kv_total : 0;
    int* const cache_cu = Paged ? cu_kv : nullptr;

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {max_seqlen_q, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {new_max, new_total, new_cu, nullptr},
             {cache_max, cache_total, cache_cu, nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
            nullptr,
            {},
            nullptr,
            {},
            nullptr,
            {},
            q_descale,
            k_descale,
            v_descale,
            32,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            sm_sink,
        },
        {
            softmax_scale,
            page_table_ptr,
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
        },
        {},
        hw_info,
    };

    if constexpr (!Sink) {
      if (num_kv_splits > 1) {
        TORCH_CHECK(
            num_kv_splits <= SplitKernel::max_num_kv_splits,
            "flash_attentionXe35: num_kv_splits exceeds the split-KV kernel limit");
        const auto shape = arguments.kernel.shape;
        const int64_t total_out_elts = static_cast<int64_t>(q_dense.size(0)) * num_heads_q * head_dim * num_kv_splits;
        const int64_t total_lse_elts = static_cast<int64_t>(q_dense.size(0)) * num_heads_q * num_kv_splits;
        auto& cache = detail::get_split_kv_cache();
        cache.ensure(total_out_elts, total_lse_elts, out_dense.options(), q_dense.options().dtype(at::kFloat));
        auto temp_out = cache.temp_out.narrow(0, 0, total_out_elts);
        auto exp_sums = cache.exp_sums.narrow(0, 0, total_lse_elts);
        auto max_logits = cache.max_logits.narrow(0, 0, total_lse_elts);

        auto split_stride_o = cutlass::make_cute_packed_stride(
            typename SplitKernel::StrideO{},
            cute::make_shape(max_seqlen_q, head_dim, num_heads_q * num_kv_splits, batch));
        auto split_stride_lse = cutlass::make_cute_packed_stride(
            typename SplitKernel::StrideO{}, cute::make_shape(max_seqlen_q, num_kv_splits, num_heads_q, batch));

        typename SplitKernel::Arguments split_arguments{
            {shape,
             static_cast<const ElementInput*>(q_dense.data_ptr()),  // Q
             stride_q,                                              // dQ
             static_cast<const ElementInput*>(k_pool.data_ptr()),   // K
             stride_k,                                              // dK
             static_cast<const ElementInput*>(v_pool.data_ptr()),   // V
             stride_v,                                              // dV
             static_cast<ElementOutput*>(temp_out.data_ptr()),      // O
             stride_o,                                              // dO
             nullptr,                                               // scaleQ
             {},                                                    // dScaleQ
             nullptr,                                               // scaleK
             {},                                                    // dScaleK
             nullptr,                                               // scaleV
             {},                                                    // dScaleV
             q_descale,                                             // scale_q
             k_descale,                                             // scale_k
             v_descale,                                             // scale_v
             32,                                                    // group_size
             static_cast<const ElementInput*>(k_pool.data_ptr()),   // K_cache
             stride_k,                                              // dK_cache
             static_cast<const ElementInput*>(v_pool.data_ptr()),   // V_cache
             stride_v,                                              // dV_cache
             nullptr,                                               // sm_sink
             static_cast<ElementOutput*>(temp_out.data_ptr()),      // Oaccum
             split_stride_o,                                        // dOaccum
             exp_sums.data_ptr<float>(),                            // exp_sums
             split_stride_lse,                                      // dExp_sums
             max_logits.data_ptr<float>(),                          // max_logits
             split_stride_lse},                                     // dMax_logits
            arguments.mainloop,
            arguments.epilogue,
            hw_info,
            num_kv_splits};
        typename ReduceKernel::Arguments reduce_arguments{
            {shape,
             static_cast<ElementOutput*>(out_dense.data_ptr()),
             stride_o,
             static_cast<const ElementOutput*>(temp_out.data_ptr()),
             stride_o,
             exp_sums.data_ptr<float>(),
             stride_o,
             max_logits.data_ptr<float>(),
             stride_o,
             window_size_left,
             v_descale},
            hw_info,
            num_kv_splits};

        typename SplitKernel::Params split_params = SplitKernel::to_underlying_arguments(split_arguments, nullptr);
        typename ReduceKernel::Params reduce_params = ReduceKernel::to_underlying_arguments(reduce_arguments, nullptr);
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;
        auto launch_split = [&](auto kernel_tag, auto params) {
          using Kernel = decltype(kernel_tag);
          compat::dim3 const block = Kernel::get_block_shape();
          compat::dim3 const grid = Kernel::get_grid_shape(params);
          compat::experimental::launch_properties launch_props{
              syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
          compat::experimental::kernel_properties kernel_props{
              syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
          compat::experimental::launch_policy policy{
              compat::dim3(grid.x, grid.y, grid.z),
              compat::dim3(block.x, block.y, block.z),
              launch_props,
              kernel_props};
          return compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel, false>(
              policy, at::xpu::getCurrentXPUStream().queue(), params);
        };
        launch_split(SplitKernel{}, split_params);
        launch_split(ReduceKernel{}, reduce_params);
        return;
      }
    }

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid cutlass FMHA arguments");

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    // Zero-init split-reduction workspace on the torch stream.
    at::Tensor workspace = at::zeros({static_cast<long>(workspace_size)}, q_dense.options().dtype(at::kByte));
    auto* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& queue = xe35_fmha::torch_stream_queue(q_dense);
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(policy, queue, params);
#endif
  }
};

template <class Element>
struct Xe3FmhaDecodeSplitKvConfig {
  using ElementInput = Element;
  using ElementOutput = Element;
  using ElementScale = float;
  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  // Decode (one query per request): all GQA heads of a KV head share one
  // workgroup and the cache is split across `num_kv_splits` workgroups.  See
  // xe_fmha_fwd_decode_splitkv_kernel.hpp.
  template <
      bool LocalMask,
      int PipelineStages,
      int FixedPageSize,
      int GrfSize,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut>
  static void run_decode_splitkv(
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
      int max_seqlen_kv_cache,
      int head_dim,
      int page_size,
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      int num_kv_splits) {
    using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
    constexpr int SGTileQ = get<0>(TileShapeQK{});
    static_assert(SGTileQ == 8, "decode GQA tile packs the query heads of one KV head into one DPAS M block");
    using MMAOperation = XE_DPAS_TT<8, float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
    static_assert(get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        cutlass::fmha::XeDefault<PipelineStages>,
        /*Causal=*/false,
        LocalMask,
        /*BlockScale=*/false,
        /*F8kvF16mma=*/false,
        /*PerTensorScale=*/false,
        /*CachedKV=*/true,
        /*PagedKV=*/true,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void,
        -2,
        -2,
        FixedPageSize>;
    using CollectiveEpilogue = cutlass::fmha::collective::
        FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, false, XeStoreCachePolicy::kWT_WB_WB>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFMHADecodeSplitKvTileScheduler;
    using FMHAKernel = cutlass::fmha::kernel::
        XeFMHAFwdDecodeSplitKvKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler, GrfSize>;

    typename FMHAKernel::StrideQ stride_q{
        static_cast<int>(q_dense.stride(0)), {}, static_cast<int>(q_dense.stride(1)), 0};
    typename FMHAKernel::StrideK stride_k{
        static_cast<int>(k_pool.stride(0)), {}, static_cast<int>(k_pool.stride(1)), 0};
    typename FMHAKernel::StrideV stride_v{
        {}, static_cast<int>(v_pool.stride(0)), static_cast<int>(v_pool.stride(1)), 0};
    typename FMHAKernel::StrideO stride_o{
        static_cast<int>(out_dense.stride(0)), {}, static_cast<int>(out_dense.stride(1)), 0};

    int* counters = nullptr;
    void* partials = nullptr;
    if (num_kv_splits > 1) {
      decode_splitkv_workspace(
          q_dense, batch * num_heads_kv, num_kv_splits, FMHAKernel::partial_bytes_per_split(), &counters, &partials);
    }

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {1, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {0, 0, nullptr, nullptr},
             {max_seqlen_kv_cache, static_cast<int>(k_pool.size(0)), cu_seqlens_k_cache.data_ptr<int>(), nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
        },
        {},
        num_kv_splits,
        static_cast<typename FMHAKernel::ElementA*>(partials),
        counters,
    };

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid decode split-KV arguments");
    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<GrfSize>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& queue = xe35_fmha::torch_stream_queue(q_dense);
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(policy, queue, params);
#endif
  }
};

// Decode dispatch for the split-KV GQA kernel.  Returns false when the request
// is not a plain single-token decode this kernel covers.
template <int HeadDim>
bool dispatch_xe3_decode_splitkv(
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
    float softmax_scale,
    bool is_local,
    int window_size_left,
    int window_size_right,
    bool use_sink) {
  if (use_sink) return false;
  if (max_seqlen_q != 1 || q_dense.size(0) != batch) return false;
  if (num_heads_kv <= 0 || num_heads_q % num_heads_kv != 0) return false;
  const int gqa_group = num_heads_q / num_heads_kv;
  if (gqa_group < 1 || gqa_group > 8) return false;
  if (page_size != 64 && page_size != 128) return false;

  // Upper bound on the cached tiles any request can touch: the page table
  // width bounds the cache length, and the window bounds the local layers.
  constexpr int kBlkK = 64;
  int max_tiles = (num_pages_per_seq * page_size + kBlkK - 1) / kBlkK;
  if (is_local && window_size_left >= 0) {
    max_tiles = std::min(max_tiles, (window_size_left + 1 + kBlkK - 1) / kBlkK + 1);
  }
  const int num_kv_splits = choose_decode_kv_splits(batch * num_heads_kv, std::max(1, max_tiles));
  constexpr int grf = 512;

  using ShapeQK = Shape<_8, Int<kBlkK>, _32>;
  using ShapePV = Shape<_8, _32, Int<kBlkK>>;
  using ShapeOut = Shape<_8, Int<HeadDim>>;

#define XE35_DECODE_SPLITKV_RUN(LOCAL, FIXED_PAGE, GRF)                                   \
  Xe3FmhaDecodeSplitKvConfig<cutlass::bfloat16_t>::                                       \
      template run_decode_splitkv<LOCAL, 2, FIXED_PAGE, GRF, ShapeQK, ShapePV, ShapeOut>( \
          q_dense,                                                                        \
          k_pool,                                                                         \
          v_pool,                                                                         \
          page_table,                                                                     \
          num_pages_per_seq,                                                              \
          cu_seqlens_q,                                                                   \
          cu_seqlens_k_cache,                                                             \
          out_dense,                                                                      \
          batch,                                                                          \
          num_heads_q,                                                                    \
          num_heads_kv,                                                                   \
          max_seqlen_kv_cache,                                                            \
          HeadDim,                                                                        \
          page_size,                                                                      \
          softmax_scale,                                                                  \
          window_size_left,                                                               \
          window_size_right,                                                              \
          num_kv_splits)

#define XE35_DECODE_SPLITKV_RUN_GRF(LOCAL, FIXED_PAGE) \
  do {                                                 \
    if (grf == 256)                                    \
      XE35_DECODE_SPLITKV_RUN(LOCAL, FIXED_PAGE, 256); \
    else                                               \
      XE35_DECODE_SPLITKV_RUN(LOCAL, FIXED_PAGE, 512); \
  } while (false)

  if (is_local) {
    if (page_size == kBlkK)
      XE35_DECODE_SPLITKV_RUN_GRF(true, kBlkK);
    else
      XE35_DECODE_SPLITKV_RUN_GRF(true, -1);
  } else {
    if (page_size == kBlkK)
      XE35_DECODE_SPLITKV_RUN_GRF(false, kBlkK);
    else
      XE35_DECODE_SPLITKV_RUN_GRF(false, -1);
  }
#undef XE35_DECODE_SPLITKV_RUN_GRF
#undef XE35_DECODE_SPLITKV_RUN
  return true;
}

template <int HeadDim, bool UseDecode, bool Paged, class Element = cutlass::bfloat16_t, class OutputElement = Element>
void dispatch_xe3_kernel(
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
    int num_kv_splits = 1) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;

  // Paged is a compile-time parameter so each translation unit only instantiates
  // one KV-layout variant; paged and non-paged are generated as separate .cpp.
#define XE35_INVOKE(CAUSAL, LOCAL, SINKV, STAGES, GQA, QK, PV, OUT, SGL, SINKPTR)                    \
  Xe3FmhaConfig<Element, OutputElement>::template run<                                               \
      CAUSAL,                                                                                        \
      LOCAL,                                                                                         \
      SINKV,                                                                                         \
      STAGES,                                                                                        \
      GQA,                                                                                           \
      Paged,                                                                                         \
      QK,                                                                                            \
      PV,                                                                                            \
      OUT,                                                                                           \
      SGL,                                                                                           \
      -2,                                                                                            \
      -2,                                                                                            \
      -1,                                                                                            \
      (UseDecode || !std::is_same_v<Element, cutlass::bfloat16_t> ? XeStoreCachePolicy::kWT_WB_WB    \
                                                                  : XeStoreCachePolicy::kWB_WB_UC)>( \
      q_dense,                                                                                       \
      k_pool,                                                                                        \
      v_pool,                                                                                        \
      page_table,                                                                                    \
      num_pages_per_seq,                                                                             \
      cu_seqlens_q,                                                                                  \
      cu_seqlens_k_cache,                                                                            \
      out_dense,                                                                                     \
      batch,                                                                                         \
      num_heads_q,                                                                                   \
      num_heads_kv,                                                                                  \
      max_seqlen_q,                                                                                  \
      max_seqlen_kv_cache,                                                                           \
      HeadDim,                                                                                       \
      page_size,                                                                                     \
      q_descale,                                                                                     \
      k_descale,                                                                                     \
      v_descale,                                                                                     \
      softmax_scale,                                                                                 \
      window_size_left,                                                                              \
      window_size_right,                                                                             \
      SINKPTR,                                                                                       \
      num_kv_splits)

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                     \
  do {                                                                             \
    if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {                  \
      if (use_sink)                                                                \
        XE35_INVOKE(CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL, sm_sink);  \
      else                                                                         \
        XE35_INVOKE(CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL, nullptr); \
    } else {                                                                       \
      TORCH_CHECK(!use_sink, "flash_attentionXe35: FP8 does not support sinks");   \
      XE35_INVOKE(CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL, nullptr);   \
    }                                                                              \
  } while (false)

  if constexpr (
      UseDecode && Paged && (HeadDim == 64 || HeadDim == 128) && std::is_same_v<Element, cutlass::bfloat16_t> &&
      std::is_same_v<OutputElement, cutlass::bfloat16_t>) {
    if (dispatch_xe3_decode_splitkv<HeadDim>(
            q_dense,
            k_pool,
            v_pool,
            page_table,
            num_pages_per_seq,
            cu_seqlens_q,
            cu_seqlens_k_cache,
            out_dense,
            batch,
            num_heads_q,
            num_heads_kv,
            max_seqlen_q,
            max_seqlen_kv_cache,
            page_size,
            softmax_scale,
            is_local,
            window_size_left,
            window_size_right,
            use_sink)) {
      return;
    }
  }

  if constexpr (HeadDim == 256) {
    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _256>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  if constexpr (HeadDim == 128) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local) {
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else if (is_causal) {
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else {
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    }
    return;
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local) {
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        } else {
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        }
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local) {
      if constexpr (
          Paged && std::is_same_v<Element, cutlass::bfloat16_t> && std::is_same_v<OutputElement, cutlass::bfloat16_t>) {
        using LocalShapeQK = Shape<_64, _64, _32>;
        using LocalShapePV = Shape<_64, _32, _64>;
        using LocalShapeOut = Shape<_64, _64>;
        using LocalSubgroupLayoutQK = Layout<Shape<_4, _1, _1>>;
#define XE35_LOCAL(FIXED_LEFT, FIXED_RIGHT, FIXED_PAGE, STAGES) \
  Xe3FmhaConfig<Element, OutputElement>::template run<          \
      false,                                                    \
      true,                                                     \
      false,                                                    \
      STAGES,                                                   \
      false,                                                    \
      Paged,                                                    \
      LocalShapeQK,                                             \
      LocalShapePV,                                             \
      LocalShapeOut,                                            \
      LocalSubgroupLayoutQK,                                    \
      FIXED_LEFT,                                               \
      FIXED_RIGHT,                                              \
      FIXED_PAGE,                                               \
      XeStoreCachePolicy::kWB_WB_UC>(                           \
      q_dense,                                                  \
      k_pool,                                                   \
      v_pool,                                                   \
      page_table,                                               \
      num_pages_per_seq,                                        \
      cu_seqlens_q,                                             \
      cu_seqlens_k_cache,                                       \
      out_dense,                                                \
      batch,                                                    \
      num_heads_q,                                              \
      num_heads_kv,                                             \
      max_seqlen_q,                                             \
      max_seqlen_kv_cache,                                      \
      HeadDim,                                                  \
      page_size,                                                \
      q_descale,                                                \
      k_descale,                                                \
      v_descale,                                                \
      softmax_scale,                                            \
      window_size_left,                                         \
      window_size_right,                                        \
      nullptr)
        if (!use_sink && window_size_left >= 0 && window_size_right == 0 && (page_size == 64 || page_size == 128)) {
          if (window_size_left == 2047) {
            if (page_size == 64)
              XE35_LOCAL(2047, 0, 64, 1);
            else
              XE35_LOCAL(2047, 0, 128, 1);
          } else if (page_size == 64) {
            XE35_LOCAL(-2, -2, 64, 1);
          } else {
            XE35_LOCAL(-2, -2, 128, 1);
          }
        } else {
          XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
        }
#undef XE35_LOCAL
      } else {
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      }
    } else if (is_causal) {
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else {
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    }
    return;
  }

#undef XE35_RUN
#undef XE35_INVOKE

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64, 128, and 256 are supported in Xe3 cutlass path");
}

}  // namespace xe35_fmha

using Xe35DispatchFn = void(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    bool,
    bool,
    int,
    int,
    const cutlass::bfloat16_t*,
    bool,
    int);
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <optional>
#include <vector>

#include "comm/common.h"
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/sycl_event_manager.hpp>
#endif
#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_epilogue.hpp"
#include "kernels/flash_attention_v2/xe3/collective/xe_fmha_fwd_mainloop.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_fmha_fwd_kernel.hpp"
#include "kernels/flash_attention_v2/xe3/kernel/xe_tile_scheduler.hpp"

#define SYCL_INTEL_TARGET 35

using namespace cute;

namespace {
sycl::queue& xe35_profiling_queue() {
  static sycl::queue queue = [] {
    sycl::queue torch_queue = at::xpu::getCurrentXPUStream().queue();
    return sycl::queue(torch_queue.get_context(), torch_queue.get_device(), sycl::property::queue::enable_profiling{});
  }();
  return queue;
}
}  // namespace

template <class Element>
struct Xe3FmhaConfig {
  using ElementInput = Element;
  using ElementOutput = Element;
  using ElementScale = float;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  template <
      bool Causal,
      bool LocalMask,
      bool Sink,
      int PipelineStages,
      bool GqaFusion,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut,
      class SubgroupLayoutQK>
  static void
  run(const at::Tensor& q_dense,
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
      int head_dim,
      int page_size,
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink) {
    static constexpr bool BlockScale = false;
    static constexpr bool F8kvF16mma = false;
    static constexpr bool PerTensorScale = false;
    static constexpr bool CachedKV = true;
    static constexpr bool PagedKV = true;

    constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q dimension");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        LocalMask,
        BlockScale,
        F8kvF16mma,
        PerTensorScale,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void>;

    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion>;
    using FMHAKernel =
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    const int q_seq_stride = static_cast<int>(q_dense.stride(0));
    const int q_head_stride = static_cast<int>(q_dense.stride(1));
    const int kv_seq_stride = static_cast<int>(k_pool.stride(0));
    const int kv_head_stride = static_cast<int>(k_pool.stride(1));
    const int o_seq_stride = static_cast<int>(out_dense.stride(0));
    const int o_head_stride = static_cast<int>(out_dense.stride(1));
    typename FMHAKernel::StrideQ stride_q{q_seq_stride, {}, q_head_stride, 0};
    typename FMHAKernel::StrideK stride_k{kv_seq_stride, {}, kv_head_stride, 0};
    typename FMHAKernel::StrideV stride_v{{}, kv_seq_stride, kv_head_stride, 0};
    typename FMHAKernel::StrideO stride_o{o_seq_stride, {}, o_head_stride, 0};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {max_seqlen_q, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {0, 0, nullptr, nullptr},
             {max_seqlen_kv_cache, static_cast<int>(k_pool.size(0)), cu_seqlens_k_cache.data_ptr<int>(), nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
            nullptr,
            {},
            nullptr,
            {},
            nullptr,
            {},
            1.0f,
            1.0f,
            1.0f,
            32,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            sm_sink,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
        },
        {},
        hw_info,
    };

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid cutlass FMHA arguments");

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    at::Tensor workspace = at::empty({static_cast<long>(workspace_size)}, q_dense.options().dtype(at::kByte));

    auto* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
    auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "flash_attentionXe35: FMHA workspace initialization failed");

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& profiling_queue = xe35_profiling_queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, profiling_queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(
        policy, profiling_queue, params);
#endif
  }
};

template <int HeadDim, bool UseDecode>
void dispatch_xe3_kernel(
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
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits = 1) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;
  const bool use_decode = UseDecode;

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                                               \
  do {                                                                                                       \
    if (use_sink) {                                                                                          \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL>(  \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          sm_sink);                                                                                          \
    } else {                                                                                                 \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL>( \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          nullptr);                                                                                          \
    }                                                                                                        \
  } while (false)

  if constexpr (HeadDim == 128) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local)
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        else
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64 and 128 are supported in Xe3 cutlass path");
}

#undef XE35_RUN

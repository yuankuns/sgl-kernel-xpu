#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

namespace at::native::xpu {
namespace {
constexpr int kWorkgroup = 256;

template <typename Index>
class MuseSmallPermutationKernel;

template <typename Index>
void launch_small_permutation(
    const Index* topk_ids,
    int32_t* padded_counts,
    int32_t* perm_src,
    int32_t* expanded_to_permuted,
    int32_t* scale_map,
    int num_pairs,
    int num_experts,
    int num_rows,
    int topk,
    int alignment,
    sycl::queue& queue) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> counts(sycl::range<1>(num_experts), cgh);
    sycl::local_accessor<int32_t, 1> starts(sycl::range<1>(num_experts), cgh);
    sycl::local_accessor<int32_t, 1> padded(sycl::range<1>(num_experts), cgh);
    sycl::local_accessor<int32_t, 1> expert_ids(sycl::range<1>(kWorkgroup), cgh);
    sycl::local_accessor<int32_t, 1> total_rows(sycl::range<1>(1), cgh);
    cgh.parallel_for<MuseSmallPermutationKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(kWorkgroup), sycl::range<1>(kWorkgroup)), [=](sycl::nd_item<1> item) {
          const int lane = static_cast<int>(item.get_local_id(0));
          for (int expert = lane; expert < num_experts; expert += kWorkgroup) {
            counts[expert] = 0;
          }
          if (lane < num_pairs) {
            expert_ids[lane] = static_cast<int32_t>(topk_ids[lane]);
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (lane < num_pairs) {
            sycl::atomic_ref<
                int32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::local_space>
                count(counts[expert_ids[lane]]);
            count.fetch_add(1);
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (num_experts <= kWorkgroup) {
            const int rows = lane < num_experts ? ((counts[lane] + alignment - 1) / alignment) * alignment : 0;
            const int base = sycl::exclusive_scan_over_group(item.get_group(), rows, sycl::plus<int>());
            if (lane < num_experts) {
              starts[lane] = base;
              padded[lane] = rows;
              padded_counts[lane] = rows;
            }
            if (lane == kWorkgroup - 1) total_rows[0] = base + rows;
          } else {
            // Power-of-two expert counts above 256 assign two or four adjacent
            // experts per lane. A workgroup scan replaces the serial 1024-expert
            // prefix and keeps starts in expert order for stable routing.
            const int per_lane = num_experts / kWorkgroup;
            const int first = lane * per_lane;
            int sum = 0;
            for (int j = 0; j < per_lane; ++j) {
              const int count = counts[first + j];
              sum += ((count + alignment - 1) / alignment) * alignment;
            }
            int row = sycl::exclusive_scan_over_group(item.get_group(), sum, sycl::plus<int>());
            for (int j = 0; j < per_lane; ++j) {
              const int expert = first + j;
              const int rows = ((counts[expert] + alignment - 1) / alignment) * alignment;
              starts[expert] = row;
              padded[expert] = rows;
              padded_counts[expert] = rows;
              row += rows;
            }
            if (lane == kWorkgroup - 1) total_rows[0] = row;
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int row = total_rows[0] + lane; row < num_rows; row += kWorkgroup) {
            perm_src[row] = 0;
            scale_map[2 * row] = row;
            scale_map[2 * row + 1] = 0;
          }
          if (lane >= num_pairs) return;
          const int expert = expert_ids[lane];
          int rank = 0;
          int earlier = 0;
          for (; earlier + 3 < lane; earlier += 4) {
            rank += (expert_ids[earlier] == expert) + (expert_ids[earlier + 1] == expert) +
                    (expert_ids[earlier + 2] == expert) + (expert_ids[earlier + 3] == expert);
          }
          for (; earlier < lane; ++earlier) {
            rank += expert_ids[earlier] == expert;
          }
          const int base = starts[expert];
          const int rows = padded[expert];
          const int dst = base + rank;
          expanded_to_permuted[lane] = dst;
          perm_src[dst] = lane / topk;
          scale_map[2 * dst] = base;
          scale_map[2 * dst + 1] = rows;
          if (rank == counts[expert] - 1) {
            for (int k = 1; k < alignment && dst + k < base + rows; ++k) {
              const int pad = dst + k;
              perm_src[pad] = 0;
              scale_map[2 * pad] = base;
              scale_map[2 * pad + 1] = -rows;
            }
          }
        });
  });
}
}  // namespace

void muse_mxfp4_small_permutation(
    torch::Tensor topk_ids,
    torch::Tensor padded_counts,
    torch::Tensor perm_src,
    torch::Tensor expanded_to_permuted,
    torch::Tensor scale_map,
    int64_t alignment) {
  TORCH_CHECK(
      topk_ids.is_xpu() && topk_ids.dim() == 2 && topk_ids.is_contiguous() &&
          (topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong),
      "topk_ids must be contiguous XPU int32 or int64 [tokens, topk]");
  TORCH_CHECK(
      padded_counts.is_xpu() && padded_counts.dim() == 1 && padded_counts.is_contiguous() &&
          padded_counts.scalar_type() == at::kInt,
      "padded_counts must be contiguous XPU int32 [experts]");
  TORCH_CHECK(
      perm_src.is_xpu() && perm_src.dim() == 1 && perm_src.is_contiguous() && perm_src.scalar_type() == at::kInt,
      "perm_src must be contiguous XPU int32 [padded_rows]");
  TORCH_CHECK(
      expanded_to_permuted.is_xpu() && expanded_to_permuted.dim() == 1 && expanded_to_permuted.is_contiguous() &&
          expanded_to_permuted.scalar_type() == at::kInt,
      "expanded_to_permuted must be contiguous XPU int32 [pairs]");
  TORCH_CHECK(
      scale_map.is_xpu() && scale_map.dim() == 2 && scale_map.is_contiguous() && scale_map.scalar_type() == at::kInt &&
          scale_map.size(1) == 2,
      "scale_map must be contiguous XPU int32 [padded_rows, 2]");
  const auto pairs = topk_ids.numel();
  const auto experts = padded_counts.numel();
  TORCH_CHECK(
      topk_ids.size(1) > 0 && pairs <= kWorkgroup && experts > 0 && experts <= 1024 && (experts & (experts - 1)) == 0 &&
          alignment > 0 && alignment <= 64 && expanded_to_permuted.numel() == pairs &&
          scale_map.size(0) == perm_src.numel() &&
          perm_src.numel() >= pairs + (alignment - 1) * std::min(experts, pairs),
      "invalid small permutation dimensions or alignment");
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (topk_ids.scalar_type() == at::kInt) {
    launch_small_permutation(
        topk_ids.data_ptr<int32_t>(),
        padded_counts.data_ptr<int32_t>(),
        perm_src.data_ptr<int32_t>(),
        expanded_to_permuted.data_ptr<int32_t>(),
        scale_map.data_ptr<int32_t>(),
        static_cast<int>(pairs),
        static_cast<int>(experts),
        static_cast<int>(perm_src.numel()),
        static_cast<int>(topk_ids.size(1)),
        static_cast<int>(alignment),
        queue);
  } else {
    launch_small_permutation(
        topk_ids.data_ptr<int64_t>(),
        padded_counts.data_ptr<int32_t>(),
        perm_src.data_ptr<int32_t>(),
        expanded_to_permuted.data_ptr<int32_t>(),
        scale_map.data_ptr<int32_t>(),
        static_cast<int>(pairs),
        static_cast<int>(experts),
        static_cast<int>(perm_src.numel()),
        static_cast<int>(topk_ids.size(1)),
        static_cast<int>(alignment),
        queue);
  }
}
}  // namespace at::native::xpu

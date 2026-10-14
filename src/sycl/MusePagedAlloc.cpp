#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"

namespace at::native::xpu {
namespace {

constexpr int kWorkgroup = 256;
constexpr int kDecodeWorkgroup = 32;

inline int64_t ceil_pages(int64_t tokens, int64_t page_size) {
  return (tokens + page_size - 1) / page_size;
}

template <typename LastScalar, int TilesPerReq, bool Page64>
struct MuseAllocExtendKernel {
  const int64_t* __restrict__ prefix_lens;
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int64_t batch;
  int64_t page_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / TilesPerReq;
    const int64_t tile = item.get_group(0) % TilesPerReq;
    const int64_t lane = item.get_local_id(0);
    if (req >= batch) {
      return;
    }
    const int64_t effective_page_size = Page64 ? 64 : page_size;
    int64_t output_start = 0;
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      output_start += seq_lens[i] - prefix_lens[i];
      new_page_start += ceil_pages(seq_lens[i], effective_page_size) - ceil_pages(prefix_lens[i], effective_page_size);
    }
    const int64_t prefix_len = prefix_lens[req];
    const int64_t seq_len = seq_lens[req];
    const int64_t old_boundary = ceil_pages(prefix_len, effective_page_size) * effective_page_size;
    for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len - prefix_len; offset += TilesPerReq * kWorkgroup) {
      const int64_t position = prefix_len + offset;
      int64_t location;
      if (position < old_boundary) {
        location = last_loc[req] + 1 + offset;
      } else {
        const int64_t page_offset = (position - old_boundary) / effective_page_size;
        location = free_pages[new_page_start + page_offset] * effective_page_size + position % effective_page_size;
      }
      out_indices[output_start + offset] = location;
    }
  }
};

// One launch produces the paged locations and writes the request table.
template <typename LastScalar, int TilesPerReq, bool Page64>
struct MuseAllocExtendAndWriteKernel {
  const int64_t* __restrict__ prefix_lens;
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ req_pool_indices;
  const uint64_t* __restrict__ prefix_pointers;
  int64_t batch;
  int64_t page_size;
  int64_t row_stride;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / TilesPerReq;
    const int64_t tile = item.get_group(0) % TilesPerReq;
    const int64_t lane = item.get_local_id(0);
    if (req >= batch) return;
    const int64_t effective_page_size = Page64 ? 64 : page_size;
    int64_t output_start = 0;
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      output_start += seq_lens[i] - prefix_lens[i];
      if constexpr (Page64) {
        // Valid prefix/sequence lengths are nonnegative; ceil(x / 64).
        new_page_start += ((seq_lens[i] + 63) >> 6) - ((prefix_lens[i] + 63) >> 6);
      } else {
        new_page_start +=
            ceil_pages(seq_lens[i], effective_page_size) - ceil_pages(prefix_lens[i], effective_page_size);
      }
    }
    const int64_t prefix_len = prefix_lens[req];
    const int64_t seq_len = seq_lens[req];
    const int64_t row = req_pool_indices[req] * row_stride;
    if constexpr (Page64) {
      // Model prefill often starts every request at zero. This branch is
      // uniform within a request and avoids prefix-table reads and the
      // old-page test in the page64 hot loop; partial prefixes still use
      // the fully general implementation below.
      if (prefix_len == 0) {
        for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len; offset += TilesPerReq * kWorkgroup) {
          const int64_t location = free_pages[new_page_start + (offset >> 6)] * 64 + (offset & 63);
          out_indices[output_start + offset] = location;
          req_to_token[row + offset] = static_cast<int32_t>(location);
        }
        return;
      }
    }
    const auto* prefix = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(prefix_pointers[req]));
    for (int64_t pos = tile * kWorkgroup + lane; pos < prefix_len; pos += TilesPerReq * kWorkgroup) {
      req_to_token[row + pos] = static_cast<int32_t>(prefix[pos]);
    }
    const int64_t old_boundary =
        Page64 ? ((prefix_len + 63) & ~int64_t{63}) : ceil_pages(prefix_len, effective_page_size) * effective_page_size;
    for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len - prefix_len; offset += TilesPerReq * kWorkgroup) {
      const int64_t position = prefix_len + offset;
      int64_t location;
      if (position < old_boundary) {
        location = last_loc[req] + 1 + offset;
      } else if constexpr (Page64) {
        location = free_pages[new_page_start + ((position - old_boundary) >> 6)] * 64 + (position & 63);
      } else {
        location = free_pages[new_page_start + (position - old_boundary) / effective_page_size] * effective_page_size +
                   position % effective_page_size;
      }
      out_indices[output_start + offset] = location;
      req_to_token[row + position] = static_cast<int32_t>(location);
    }
  }
};

template <typename LastScalar, bool Page64>
struct MuseAllocDecodeKernel {
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int64_t batch;
  int64_t page_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_global_id(0);
    if (req >= batch) {
      return;
    }
    const int64_t effective_page_size = Page64 ? 64 : page_size;
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      const int64_t seq = seq_lens[i];
      new_page_start += ceil_pages(seq, effective_page_size) - ceil_pages(seq - 1, effective_page_size);
    }
    const int64_t seq = seq_lens[req];
    const bool need_page = ceil_pages(seq, effective_page_size) > ceil_pages(seq - 1, effective_page_size);
    out_indices[req] = need_page ? free_pages[new_page_start] * effective_page_size : last_loc[req] + 1;
  }
};

void check_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.dim() == 1 && tensor.is_contiguous() && tensor.scalar_type() == at::kLong,
      name,
      " must be a contiguous XPU int64 vector");
}

void check_last_loc(const torch::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.dim() == 1 && tensor.is_contiguous() &&
          (tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong),
      "last_loc must be a contiguous XPU int32 or int64 vector");
}

template <typename LastScalar, int TilesPerReq, bool Page64>
void launch_extend(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  MuseAllocExtendKernel<LastScalar, TilesPerReq, Page64> kernel{
      prefix_lens.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      batch,
      page_size};
  sycl_kernel_submit(sycl::range<1>(batch * TilesPerReq * kWorkgroup), sycl::range<1>(kWorkgroup), queue, kernel);
}

template <typename LastScalar, bool Page64>
void dispatch_extend_tiles(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    int64_t output_per_req,
    sycl::queue& queue) {
  if (output_per_req >= 8 * kWorkgroup) {
    launch_extend<LastScalar, 16, Page64>(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  } else if (output_per_req >= 2 * kWorkgroup) {
    launch_extend<LastScalar, 4, Page64>(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  } else {
    launch_extend<LastScalar, 1, Page64>(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  }
}

template <typename LastScalar, int TilesPerReq, bool Page64>
void launch_extend_and_write(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& prefix_pointers,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  MuseAllocExtendAndWriteKernel<LastScalar, TilesPerReq, Page64> kernel{
      prefix_lens.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      req_to_token.data_ptr<int32_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      prefix_pointers.data_ptr<uint64_t>(),
      batch,
      page_size,
      req_to_token.stride(0)};
  sycl_kernel_submit(sycl::range<1>(batch * TilesPerReq * kWorkgroup), sycl::range<1>(kWorkgroup), queue, kernel);
}

template <typename LastScalar, bool Page64>
void dispatch_extend_and_write(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& prefix_pointers,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  const int64_t output_per_req = out_indices.numel() / batch;
  if (output_per_req >= 8 * kWorkgroup) {
    launch_extend_and_write<LastScalar, 16, Page64>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  } else if (output_per_req >= 2 * kWorkgroup) {
    launch_extend_and_write<LastScalar, 4, Page64>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  } else {
    launch_extend_and_write<LastScalar, 1, Page64>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  }
}

template <typename LastScalar, bool Page64>
void launch_decode(
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = seq_lens.numel();
  MuseAllocDecodeKernel<LastScalar, Page64> kernel{
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      batch,
      page_size};
  sycl_kernel_submit(
      sycl::range<1>(((batch + kDecodeWorkgroup - 1) / kDecodeWorkgroup) * kDecodeWorkgroup),
      sycl::range<1>(kDecodeWorkgroup),
      queue,
      kernel);
}

}  // namespace

void muse_alloc_extend(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  check_vector(prefix_lens, "prefix_lens");
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  TORCH_CHECK(
      page_size > 0 && seq_lens.numel() == prefix_lens.numel() && last_loc.numel() == prefix_lens.numel(),
      "invalid batch or page_size");
  const int64_t batch = prefix_lens.numel();
  if (batch == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t output_per_req = out_indices.numel() / batch;
  // The serving model uses page size 64; specialize its division and modulo.
  if (last_loc.scalar_type() == at::kInt) {
    if (page_size == 64) {
      dispatch_extend_tiles<int32_t, true>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    } else {
      dispatch_extend_tiles<int32_t, false>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    }
  } else {
    if (page_size == 64) {
      dispatch_extend_tiles<int64_t, true>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    } else {
      dispatch_extend_tiles<int64_t, false>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    }
  }
}

void muse_alloc_extend_and_write(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    int64_t page_size) {
  check_vector(prefix_lens, "prefix_lens");
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  check_vector(req_pool_indices, "req_pool_indices");
  TORCH_CHECK(
      prefix_pointers.is_xpu() && prefix_pointers.dim() == 1 && prefix_pointers.is_contiguous() &&
          prefix_pointers.scalar_type() == at::kUInt64,
      "prefix_pointers must be a contiguous XPU uint64 vector");
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dim() == 2 && req_to_token.is_contiguous() &&
          req_to_token.scalar_type() == at::kInt,
      "req_to_token must be a contiguous XPU int32 matrix");
  const int64_t batch = prefix_lens.numel();
  TORCH_CHECK(
      page_size > 0 && seq_lens.numel() == batch && last_loc.numel() == batch && req_pool_indices.numel() == batch &&
          prefix_pointers.numel() == batch,
      "invalid batch or page_size");
  if (!batch) return;
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (last_loc.scalar_type() == at::kInt) {
    if (page_size == 64) {
      dispatch_extend_and_write<int32_t, true>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    } else {
      dispatch_extend_and_write<int32_t, false>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    }
  } else {
    if (page_size == 64) {
      dispatch_extend_and_write<int64_t, true>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    } else {
      dispatch_extend_and_write<int64_t, false>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    }
  }
}

void muse_alloc_decode(
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  TORCH_CHECK(
      page_size > 0 && last_loc.numel() == seq_lens.numel() && out_indices.numel() == seq_lens.numel(),
      "invalid batch or page_size");
  const int64_t batch = seq_lens.numel();
  if (batch == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (last_loc.scalar_type() == at::kInt) {
    if (page_size == 64) {
      launch_decode<int32_t, true>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    } else {
      launch_decode<int32_t, false>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    }
  } else {
    if (page_size == 64) {
      launch_decode<int64_t, true>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    } else {
      launch_decode<int64_t, false>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    }
  }
}

}  // namespace at::native::xpu

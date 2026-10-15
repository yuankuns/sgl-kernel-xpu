#include <torch/all.h>

namespace at::native::xpu {
void write_req_to_token_pool(
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor extend_lens,
    torch::Tensor out_cache_loc);
void paged_alloc_extend(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size);
void paged_alloc_extend_and_write(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    int64_t page_size);
void paged_alloc_decode(
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size);
}  // namespace at::native::xpu

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "write_req_to_token_pool(Tensor(a!) req_to_token, Tensor req_pool_indices, Tensor prefix_pointers, "
      "Tensor prefix_lens, Tensor seq_lens, Tensor extend_lens, Tensor out_cache_loc) -> ()");
  m.impl("write_req_to_token_pool", torch::kXPU, &at::native::xpu::write_req_to_token_pool);
  m.def(
      "paged_alloc_extend(Tensor prefix_lens, Tensor seq_lens, Tensor last_loc, Tensor free_pages, Tensor! "
      "out_indices, int page_size) -> ()");
  m.impl("paged_alloc_extend", torch::kXPU, &at::native::xpu::paged_alloc_extend);
  m.def(
      "paged_alloc_extend_and_write(Tensor prefix_lens, Tensor seq_lens, Tensor last_loc, Tensor free_pages, "
      "Tensor(a!) out_indices, Tensor(b!) req_to_token, Tensor req_pool_indices, Tensor prefix_pointers, int "
      "page_size) -> ()");
  m.impl("paged_alloc_extend_and_write", torch::kXPU, &at::native::xpu::paged_alloc_extend_and_write);
  m.def(
      "paged_alloc_decode(Tensor seq_lens, Tensor last_loc, Tensor free_pages, Tensor! out_indices, int page_size) -> "
      "()");
  m.impl("paged_alloc_decode", torch::kXPU, &at::native::xpu::paged_alloc_decode);
}

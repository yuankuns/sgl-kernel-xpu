#include <torch/all.h>

namespace at::native::xpu {
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

void paged_alloc_extend_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  paged_alloc_extend(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size);
}

void paged_alloc_extend_and_write_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    int64_t page_size) {
  paged_alloc_extend_and_write(prefix_lens, seq_lens, last_loc, free_pages, out_indices, req_to_token, req_pool_indices, prefix_pointers, page_size);
}

void paged_alloc_decode_xe35(
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  paged_alloc_decode(seq_lens, last_loc, free_pages, out_indices, page_size);
}

}

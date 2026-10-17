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

void write_req_to_token_pool_xe35(
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor extend_lens,
    torch::Tensor out_cache_loc) {
  write_req_to_token_pool(req_to_token, req_pool_indices, prefix_pointers, prefix_lens, seq_lens, extend_lens, out_cache_loc);
}

}

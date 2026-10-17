#include <torch/all.h>

namespace at::native::xpu {
std::tuple<torch::Tensor, torch::Tensor> vocab_mask(
    torch::Tensor input,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end);
void muse_vocab_parallel_embedding(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end);

std::tuple<torch::Tensor, torch::Tensor> vocab_mask_xe35(
    torch::Tensor input,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end) {
  return vocab_mask(input, org_start, org_end, num_org_padding, added_start, added_end);
}

void vocab_parallel_embedding_xe35(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end) {
  muse_vocab_parallel_embedding(input, weight, output, org_start, org_end, num_org_padding, added_start, added_end);
}

}

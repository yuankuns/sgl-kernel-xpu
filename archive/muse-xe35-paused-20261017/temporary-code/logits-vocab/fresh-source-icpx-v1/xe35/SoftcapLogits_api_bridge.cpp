#include <torch/all.h>

namespace at::native::xpu {
void softcap_logits(torch::Tensor logits, double cap);

void softcap_logits_xe35(torch::Tensor logits, double cap) {
  softcap_logits(logits, cap);
}

}

#include <torch/all.h>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
SGL_KERNEL_EXPORT void softcap_logits_xe35(torch::Tensor logits, double cap);

SGL_KERNEL_EXPORT void softcap_logits(torch::Tensor logits, double cap) {
  softcap_logits_xe35(logits, cap);
}

}

#include <c10/xpu/XPUStream.h>

#include "sgl_kernel/xe35/activation.h"
#include "activation.hpp"

void silu_and_mul_xe35(at::Tensor& out, at::Tensor& input) {
  input = input.contiguous();
  out = out.contiguous();

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (input.scalar_type() == at::ScalarType::Half) {
    sgl_kernel::xe35::activation::silu_and_mul_sycl<sycl::half, at::Half>(queue, input, out);
  } else {
    sgl_kernel::xe35::activation::silu_and_mul_sycl<sycl::ext::oneapi::bfloat16, at::BFloat16>(queue, input, out);
  }
  return;
}


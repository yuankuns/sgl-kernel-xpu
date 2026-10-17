#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include "sgl_kernel_ops.h"
#include "sgl_kernel/xe35/activation.h"
#include "activation.hpp"

void silu_and_mul_xe35(at::Tensor& out, at::Tensor& input) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::XPU, "Xe35 activation requires XPU input");
  TORCH_CHECK(out.device() == input.device(), "Activation output must be on the input device");
  auto input_device_index = input.get_device();
  auto input_architecture = query_device(input_device_index);
  TORCH_CHECK(input_architecture == std::make_tuple<int64_t, int64_t>(3, 5), "Xe35 activation requires input architecture (3, 5)");
  c10::DeviceGuard input_device_guard(input.device());
  input = input.contiguous();
  out = out.contiguous();

  auto stream = at::xpu::getCurrentXPUStream(input_device_index);
  auto queue = stream.queue();

  if (input.scalar_type() == at::ScalarType::Half) {
    sgl_kernel::xe35::activation::silu_and_mul_sycl_xe35<sycl::half, at::Half>(queue, input, out);
  } else {
    sgl_kernel::xe35::activation::silu_and_mul_sycl_xe35<sycl::ext::oneapi::bfloat16, at::BFloat16>(queue, input, out);
  }
  return;
}


#pragma once

#include <ATen/ATen.h>
#include "sgl_kernel_export.h"

SGL_KERNEL_EXPORT void silu_and_mul_xe35(at::Tensor& out, at::Tensor& input);

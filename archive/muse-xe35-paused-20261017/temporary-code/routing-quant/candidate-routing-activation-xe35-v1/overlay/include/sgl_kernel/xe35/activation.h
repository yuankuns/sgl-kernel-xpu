#pragma once

#include <ATen/ATen.h>

#pragma GCC visibility push(default)

void silu_and_mul_xe35(at::Tensor& out, at::Tensor& input);

#pragma GCC visibility pop

#pragma once

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <sycl/sycl.hpp>

#include "sycl/MemoryAccess.h"
#include "sycl/SYCLHelpers.h"
#include "sycl/Utils.h"

namespace sgl_kernel::xe35::activation {

template <typename scalar_t, typename accscalar_t>
struct silu_mul_dpcpp_functor_xe35 {
  scalar_t operator()(scalar_t activation_value, scalar_t multiplication_value) const {
    return (accscalar_t(activation_value)) / (1.0f + expf(accscalar_t(-activation_value))) * accscalar_t(multiplication_value);
  }
};

template <typename scalar_t, typename func_t, int VectorWidth>
struct op_and_mul_functor_xe35 {
  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = at::opmath_type<scalar_t>;
    int64_t offset = item.get_local_linear_id();
    int64_t step = item.get_local_range(0);
    int64_t token_id = item.get_group(0);
    func_t activation_function;
    int64_t bound = dim / VectorWidth;
    for (int64_t vector_index = offset; vector_index < bound; vector_index += step) {
      auto unary_val = reinterpret_cast<aligned_vector_loop<scalar_t, VectorWidth>*>(input_ptr)[token_id * bound * 2 + vector_index];
      auto mul_val = reinterpret_cast<aligned_vector_loop<scalar_t, VectorWidth>*>(input_ptr)[token_id * bound * 2 + vector_index + bound];
#pragma unroll
      for (int vector_index = 0; vector_index < VectorWidth; ++vector_index) {
        auto activation_value = unary_val[vector_index], multiplication_value = mul_val[vector_index];
        unary_val[vector_index] = activation_function(unary_val[vector_index], mul_val[vector_index]);
      }
      reinterpret_cast<aligned_vector_loop<scalar_t, VectorWidth>*>(output_ptr)[token_id * bound + vector_index] = unary_val;
    }
  }

  scalar_t* input_ptr;
  scalar_t* output_ptr;
  int64_t num_;
  int64_t dim;
};

#define SGL_XE35_SILU_VEC_LAUNCH(KERNEL, VectorWidth)                                                  \
  case VectorWidth: {                                                                    \
    op_and_mul_functor_xe35<T_to, KERNEL<T_to, at::opmath_type<T_from>>, VectorWidth> kfn = { \
        .input_ptr = _input, .output_ptr = _out, .num_ = numel, .dim = dim};   \
    sycl_kernel_submit(num_group* wg_size, wg_size, kernel_queue, kfn);                   \
    break;                                                                     \
  }

template <typename ConfigScalar = float>
void get_config_xe35(
    const at::Tensor& input,
    const at::Tensor& out,
    int64_t& numel,
    int64_t& dim,
    int64_t& wg_size,
    int64_t& num_group,
    int& vec_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  numel = out.numel();
  dim = out.size(-1);
  int64_t tokens = numel / dim;
  wg_size = std::min(dim, max_wg_size);
  num_group = tokens;

  vec_size = sizeof(float) * 4 / sizeof(ConfigScalar);
  while ((vec_size >> 1) * wg_size >= dim) {
    vec_size = vec_size >> 1;
  }
  if (dim % vec_size != 0) vec_size = 1;
}

template <typename T_to = float, typename T_from = float>
void silu_and_mul_sycl_xe35(sycl::queue& kernel_queue, at::Tensor& input, at::Tensor& out) {
  auto _input = reinterpret_cast<T_to*>(input.data_ptr<T_from>());
  auto _out = reinterpret_cast<T_to*>(out.data_ptr<T_from>());

  int64_t numel;
  int64_t dim;
  int64_t wg_size;
  int64_t num_group;
  int vec_size;
  get_config_xe35<T_to>(input, out, numel, dim, wg_size, num_group, vec_size);

  using accscalar_t = at::opmath_type<T_to>;
  switch (vec_size) {
    SGL_XE35_SILU_VEC_LAUNCH(silu_mul_dpcpp_functor_xe35, 1);
    SGL_XE35_SILU_VEC_LAUNCH(silu_mul_dpcpp_functor_xe35, 2);
    SGL_XE35_SILU_VEC_LAUNCH(silu_mul_dpcpp_functor_xe35, 4);
    SGL_XE35_SILU_VEC_LAUNCH(silu_mul_dpcpp_functor_xe35, 8);
    SGL_XE35_SILU_VEC_LAUNCH(silu_mul_dpcpp_functor_xe35, 16);
    default:
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
  }

  return;
}


#undef SGL_XE35_SILU_VEC_LAUNCH

}

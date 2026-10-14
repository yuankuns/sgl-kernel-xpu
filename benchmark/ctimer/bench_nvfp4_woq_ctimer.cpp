// Device-clock timing bridge for the NVFP4 WoQ and MXFP8 grouped GEMMs.
//
// PyTorch's simulator queue does not enable SYCL profiling. This extension
// temporarily installs an external profiling-enabled queue, then reports both
// CUTLASS GPU_Clock time (registered GEMM events only) and a barrier-bracketed
// device span (all device work submitted by the operator).

#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

#include <cstdio>
#include <sycl/sycl.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "sycl/kernels/moe/xe35/device/grouped_gemm_runner.hpp"

// These diagnostic helpers were emitted by the compiler used for the existing
// kernel DSOs but were removed from the simulator's current libsycl. They are
// only reached for async exceptions or range-rounding notices.
namespace sycl {
inline namespace _V1 {
namespace detail {
void reportExceptionToStream(const char* message, const char* category) {
  std::fprintf(stderr, "SYCL async exception [%s]: %s\n", category, message);
}
void reportAdjustedRange(int, size_t, size_t) {}
}  // namespace detail
}  // namespace _V1
}  // namespace sycl

void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace);

void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace);

void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace);

void mxfp4_moe_scale_to_mn_major(torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert);

namespace {

namespace policies = gpu::cutlass_kernel::grouped_gemm;
using sgl_kernel::vllm_grouped_gemm::kernel_functor;

struct XPUStreamRestore {
  c10::xpu::XPUStream previous;
  ~XPUStreamRestore() {
    c10::xpu::setCurrentXPUStream(previous);
  }
};

void launch_bf16_reference(
    const torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& rows_per_expert) {
  const int groups = static_cast<int>(b.size(0));
  const int n = static_cast<int>(b.size(1));
  const int k = static_cast<int>(b.size(2));
  const int avg_m = static_cast<int>(a.size(0) / groups);
  auto& queue = at::xpu::getCurrentXPUStream(a.get_device()).queue();

#define LAUNCH_BF16_REFERENCE(Policy) \
  kernel_functor<Policy>(             \
      queue,                          \
      a.data_ptr(),                   \
      nullptr,                        \
      b.data_ptr(),                   \
      nullptr,                        \
      nullptr,                        \
      output.data_ptr(),              \
      rows_per_expert.data_ptr(),     \
      n,                              \
      k,                              \
      groups)

  if (avg_m <= 4) {
    LAUNCH_BF16_REFERENCE(policies::moe_bf16_decode_k64_policy);
  } else if (avg_m <= 32) {
    LAUNCH_BF16_REFERENCE(policies::moe_bf16_mid_policy);
  } else {
    LAUNCH_BF16_REFERENCE(policies::moe_bf16_policy);
  }

#undef LAUNCH_BF16_REFERENCE
}

void launch_mxfp8_reference(
    const torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& rows_per_expert) {
  const int groups = static_cast<int>(b.size(0));
  const int n = static_cast<int>(b.size(1));
  const int k = static_cast<int>(b.size(2));
  const int avg_m = static_cast<int>(a.size(0) / groups);
  auto& queue = at::xpu::getCurrentXPUStream(a.get_device()).queue();

#define LAUNCH_MXFP8_REFERENCE(Policy) \
  kernel_functor<Policy>(              \
      queue,                           \
      a.data_ptr(),                    \
      scales_a.data_ptr(),             \
      b.data_ptr(),                    \
      scales_b.data_ptr(),             \
      nullptr,                         \
      output.data_ptr(),               \
      rows_per_expert.data_ptr(),      \
      n,                               \
      k,                               \
      groups)

  if (avg_m <= 4) {
    LAUNCH_MXFP8_REFERENCE(policies::moe_mxfp8_decode_policy);
  } else if (avg_m <= 32) {
    LAUNCH_MXFP8_REFERENCE(policies::moe_mxfp8_mid_policy);
  } else {
    LAUNCH_MXFP8_REFERENCE(policies::moe_mxfp8_policy);
  }

#undef LAUNCH_MXFP8_REFERENCE
}

template <class Launch>
pybind11::dict profile_on_device_queue(int device_index, int warmup, int iterations, Launch&& launch) {
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  TORCH_CHECK(iterations > 0, "iterations must be positive");

  auto previous = c10::xpu::getCurrentXPUStream(device_index);
  auto& previous_queue = previous.queue();
  sycl::queue queue{
      previous_queue.get_context(),
      previous_queue.get_device(),
      sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};
  auto profiling_stream = c10::xpu::getStreamFromExternal(&queue, device_index);
  XPUStreamRestore restore{previous};
  c10::xpu::setCurrentXPUStream(profiling_stream);

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  queue.wait_and_throw();

  GPU_Clock timer;
  timer.start();
  auto before = queue.ext_oneapi_submit_barrier();
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  auto after = queue.ext_oneapi_submit_barrier();
  after.wait_and_throw();

  const double cute_ms = static_cast<double>(timer.milliseconds()) / iterations;
  const auto before_end = before.get_profiling_info<sycl::info::event_profiling::command_end>();
  const auto after_start = after.get_profiling_info<sycl::info::event_profiling::command_start>();
  const double span_ms = static_cast<double>(after_start - before_end) * 1e-6 / iterations;

  pybind11::dict result;
  result["cute_ms"] = cute_ms;
  result["device_span_ms"] = span_ms;
  result["profiling"] = queue.has_property<sycl::property::queue::enable_profiling>();
  return result;
}

}  // namespace

pybind11::dict benchmark_nvfp4(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace,
    int warmup,
    int iterations) {
  return profile_on_device_queue(a.get_device(), warmup, iterations, [&] {
    nvfp4_woq_mxfp8_scaled_grouped_mm(output, a, b, scales_a, sfmx_b, c_b, problem_sizes, expert_offsets, workspace);
  });
}

void run_nvfp4(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  nvfp4_woq_mxfp8_scaled_grouped_mm(output, a, b, scales_a, sfmx_b, c_b, problem_sizes, expert_offsets, workspace);
}

pybind11::dict benchmark_nvfp4_bf16(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace,
    int warmup,
    int iterations) {
  return profile_on_device_queue(a.get_device(), warmup, iterations, [&] {
    nvfp4_woq_bf16_grouped_mm(output, a, b, sfmx_b, c_b, problem_sizes, expert_offsets, workspace);
  });
}

void run_nvfp4_bf16(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  nvfp4_woq_bf16_grouped_mm(output, a, b, sfmx_b, c_b, problem_sizes, expert_offsets, workspace);
}

pybind11::dict benchmark_mxfp8(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    torch::Tensor empty_ptrs,
    const torch::Tensor& zero_stride,
    const torch::Tensor& zero_layout,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace,
    int warmup,
    int iterations) {
  return profile_on_device_queue(a.get_device(), warmup, iterations, [&] {
    auto a_ptrs = empty_ptrs;
    auto b_ptrs = empty_ptrs;
    auto out_ptrs = empty_ptrs;
    auto a_scale_ptrs = empty_ptrs;
    auto b_scale_ptrs = empty_ptrs;
    fp8_blockwise_scaled_grouped_mm(
        output,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scale_ptrs,
        b_scale_ptrs,
        a,
        b,
        scales_a,
        scales_b,
        zero_stride,
        zero_stride,
        zero_stride,
        zero_layout,
        zero_layout,
        problem_sizes,
        expert_offsets,
        workspace);
  });
}

pybind11::dict benchmark_bf16(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& rows_per_expert,
    int warmup,
    int iterations) {
  return profile_on_device_queue(
      a.get_device(), warmup, iterations, [&] { launch_bf16_reference(output, a, b, rows_per_expert); });
}

pybind11::dict benchmark_mxfp8_stock(
    torch::Tensor output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& rows_per_expert,
    int warmup,
    int iterations) {
  return profile_on_device_queue(a.get_device(), warmup, iterations, [&] {
    launch_mxfp8_reference(output, a, b, scales_a, scales_b, rows_per_expert);
  });
}

void prepare_mxfp8_scales(torch::Tensor output, const torch::Tensor& input, const torch::Tensor& rows_per_expert) {
  mxfp4_moe_scale_to_mn_major(output, input, rows_per_expert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("benchmark_nvfp4", &benchmark_nvfp4);
  m.def("run_nvfp4", &run_nvfp4);
  m.def("benchmark_nvfp4_bf16", &benchmark_nvfp4_bf16);
  m.def("run_nvfp4_bf16", &run_nvfp4_bf16);
  m.def("benchmark_mxfp8", &benchmark_mxfp8);
  m.def("benchmark_bf16", &benchmark_bf16);
  m.def("benchmark_mxfp8_stock", &benchmark_mxfp8_stock);
  m.def("prepare_mxfp8_scales", &prepare_mxfp8_scales);
}

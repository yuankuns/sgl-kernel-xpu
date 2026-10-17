cmake_minimum_required(VERSION 3.19.2)
project(muse_residual_norm_source_xe35 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX})
find_package(Torch REQUIRED)
set(SGL_OPS_XPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../source-snapshot)
list(APPEND CMAKE_MODULE_PATH ${SGL_OPS_XPU_ROOT}/cmake/Modules)
set(DPCPP_SYCL_TARGET cri)
include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)
include_directories(${SGL_OPS_XPU_ROOT}/src/sycl ${SGL_OPS_XPU_ROOT}/src ${TORCH_INCLUDE_DIRS})
set(COMMON_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)
sycl_add_library(sgl-ops-sycl-FusedNormResidual ${SYCL_OFFLINE_COMPILER_FLAGS}
  ${COMMON_DEVICE_LINK_FLAGS} SHARED
  SYCL_SOURCES ${SGL_OPS_XPU_ROOT}/src/sycl/FusedNormResidual.cpp)
torch_compile_options(sgl-ops-sycl-FusedNormResidual)
target_compile_options(sgl-ops-sycl-FusedNormResidual PRIVATE ${TORCH_XPU_OPS_FLAGS})
target_link_libraries(sgl-ops-sycl-FusedNormResidual PRIVATE ${TORCH_LIBRARIES} ${SYCL_LIBRARY})
set_target_properties(sgl-ops-sycl-FusedNormResidual PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS sgl-ops-sycl-FusedNormResidual LIBRARY DESTINATION muse_residual_norm_source_xe35)

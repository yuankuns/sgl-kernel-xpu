#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
CXX="${CXX:-icx}"
CUTLASS="${CUTLASS:-$ROOT/build/_deps/repo-cutlass-sycl-src}"
if [[ ! -d "$CUTLASS/include" ]]; then
  CUTLASS=/root/libraries.ai.cutlass.internal
fi
TORCH_ROOT="$($PYTHON -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')"
TORCH_LIB="$TORCH_ROOT/lib"
PYTHON_INCLUDE="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
ABI="$($PYTHON -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
OUT="$HERE/nvfp4_ctimer_ext$EXT_SUFFIX"
HOST_OPTIONS="-fPIC -std=c++20 \
-I/opt/gfx-deps/dpcpp/include \
-I$PYTHON_INCLUDE \
-I$TORCH_ROOT/include \
-I$TORCH_ROOT/include/torch/csrc/api/include \
-I$CUTLASS/include \
-I$CUTLASS/tools/util/include \
-I$ROOT/src \
-DTORCH_EXTENSION_NAME=nvfp4_ctimer_ext \
-DCUTLASS_ENABLE_SYCL \
-DCUTLASS_SYCL_PROFILING_ENABLED \
-DSGL_PRE_SILICON \
-DSYCL_INTEL_TARGET=35 \
-D__SYCL_TARGET_INTEL_GPU_CRI__ \
-D__INTEL_LLVM_COMPILER_VERSION=20260200 \
-D_GLIBCXX_USE_CXX11_ABI=$ABI"

"$CXX" \
  -fsycl -fsycl-targets=spir64_gen \
  -fsycl-max-parallel-link-jobs=1 \
  -fsycl-host-compiler=/usr/bin/x86_64-linux-gnu-g++ \
  "-fsycl-host-compiler-options=$HOST_OPTIONS" \
  -sycl-std=2020 -shared -fPIC -O2 -std=c++20 \
  --offload-compress \
  -Xspirv-translator \
  "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
  -Xs "-device cri -internal_options -cl-intel-512-GRF-per-thread" \
  -DTORCH_EXTENSION_NAME=nvfp4_ctimer_ext \
  -DCUTLASS_ENABLE_SYCL \
  -DCUTLASS_SYCL_PROFILING_ENABLED \
  -DSGL_PRE_SILICON \
  -DSYCL_INTEL_TARGET=35 \
  -D__SYCL_TARGET_INTEL_GPU_CRI__ \
  -D__INTEL_LLVM_COMPILER_VERSION=20260200 \
  -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  -I"$PYTHON_INCLUDE" \
  -I"$TORCH_ROOT/include" \
  -I"$TORCH_ROOT/include/torch/csrc/api/include" \
  -I"$CUTLASS/include" \
  -I"$CUTLASS/tools/util/include" \
  -I"$ROOT/src" \
  "$HERE/bench_nvfp4_woq_ctimer.cpp" \
  "$ROOT/build/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so" \
  "$ROOT/build/src/libsgl-ops-sycl-blockwise_moe_mxfp8.so" \
  "$ROOT/build/src/libsgl-ops-sycl-mxfp4_moe_scale_layout.so" \
  -L"$TORCH_LIB" \
  -ltorch_python -ltorch_xpu -ltorch_cpu -ltorch -lc10_xpu -lc10 \
  -Wl,-rpath,"$TORCH_LIB" \
  -Wl,-rpath,"$ROOT/build/src" \
  -o "$OUT"

echo "built $OUT"

#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
target=/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35
exec 9>"$artifact/port-compile-audit.lock"
flock 9
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_MLA_SPARSE_FUSED USE_SYCL_JIT CMAKE_ARGS SKBUILD_CMAKE_ARGS
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
python="$artifact/target-venv/bin/python"
cmake="$artifact/target-venv/bin/cmake"
"$python" -m cmake -S "$target" -B "$artifact/port-compile-audit" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=cri \
  -DPython3_EXECUTABLE="$python" -DPYTHON3_EXECUTABLE="$python" \
  -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.10 \
  -DCMAKE_PROJECT_INCLUDE="$artifact/diagnostic_payload_output.cmake" \
  -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56
"$python" -m cmake --build "$artifact/port-compile-audit" --target \
  sgl-ops-sycl-blockwise_moe_nvfp4_woq sgl-ops-sycl-nvfp4_woq --parallel 1

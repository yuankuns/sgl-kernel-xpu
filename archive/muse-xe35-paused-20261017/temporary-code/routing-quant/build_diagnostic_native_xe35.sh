#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
export PATH="$artifact/port-venv/bin:$PATH"
export CC=gcc CXX=g++
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_SYCL_JIT USE_MLA_SPARSE_FUSED CMAKE_ARGS SKBUILD_CMAKE_ARGS
python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py --max-jobs 8 --jobs 2 --build-dir "$PWD/build" --state-dir "$artifact/build-state" -- bash -c '
  set -euo pipefail
  export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$1/port-venv/bin/python" -DPYTHON3_EXECUTABLE="$1/port-venv/bin/python" -DDPCPP_SYCL_TARGET=cri -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 -DSGL_BUILD_MEM_FILE_LIMIT_GIB=8
  cmake --build build --parallel "$BUILD_JOBS" --target sgl-ops-sycl-muse_routing_quant sgl-ops-sycl-per_token_group_quant_fp4_xe35 sgl-ops-sycl-mxfp4_moe_scale_layout
' bash "$artifact"

#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
source=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014
pin=/root/worktrees/cutlass_internal/muse-w4a4-e87870b
[[ "$(cat "$artifact/full-build-v2-stop-tree.exit")" == 0 ]] || exit 125
export PATH="$artifact/port-venv/bin:$PATH"
export CC=gcc CXX=g++ MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=2
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_SYCL_JIT USE_MLA_SPARSE_FUSED CMAKE_ARGS SKBUILD_CMAKE_ARGS
python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py --jobs 2 --max-jobs 2 --build-dir "$artifact/fresh-source-build" --state-dir "$artifact/fresh-source-build-state-v1" -- bash -c '
  set -euo pipefail
  cmake -S "$1" -B "$2/fresh-source-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$2/port-venv/bin/python" -DPYTHON3_EXECUTABLE="$2/port-venv/bin/python" -DDPCPP_SYCL_TARGET=cri -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL="$3" -DSYCL_COMPILER="$2/fresh_source_icx_monitor.sh" -DSYCL_EXECUTABLE="$2/fresh_source_icx_monitor.sh"
  cmake --build "$2/fresh-source-build" --parallel 2 --target sgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4
' bash "$source" "$artifact" "$pin"

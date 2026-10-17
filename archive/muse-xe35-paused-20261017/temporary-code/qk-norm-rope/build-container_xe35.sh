#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS=1
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_MLA_SPARSE_FUSED USE_SYCL_JIT CMAKE_ARGS SKBUILD_CMAKE_ARGS
exec python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py \
  --build-dir build --jobs 2 -- bash -c '
    export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
    python -m pip wheel --no-build-isolation --no-deps -v . \
      -w /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/wheels \
      -Ccmake.define.DPCPP_SYCL_TARGET=cri \
      -Ccmake.define.SGL_BUILD_MEM_FILE_LIMIT_GIB=8 \
      -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56
  '

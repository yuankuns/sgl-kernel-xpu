#!/usr/bin/env bash
set -euo pipefail
module="${1:?module required}"
case "$module" in mxfp4-w4a4|routing-quant) ;; *) exit 2 ;; esac
base=/root/worktrees/sgl-kernel-xpu-up
artifact="$base/muse-upstream-artifacts-20261017/$module"
[[ "$PWD" == "$base/pr/muse-$module-xe35" ]] || { echo 'Unexpected target workdir' >&2; exit 2; }
export PATH="$artifact/port-venv/bin:$PATH"
export PIP_CACHE_DIR="$artifact/pip-cache" XDG_CACHE_HOME="$artifact/cache"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export CC=gcc CXX=g++
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_SYCL_JIT USE_MLA_SPARSE_FUSED CMAKE_ARGS SKBUILD_CMAKE_ARGS
python /root/agent-skills/xpu-build-supervisor/scripts/run-incremental-build.py --max-jobs 8 --jobs 8 --reset-history --build-dir "$PWD/build" --state-dir "$artifact/build-state-full-v2-j8" -- bash -c '
  set -euo pipefail
  export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  printf "default full v2 jobs=%s inner_device_link=1 CPU64-79; no feature disable\n" "$BUILD_JOBS"
  python -m pip wheel --no-build-isolation --no-deps -v . -w "$1/wheels" -Ccmake.define.DPCPP_SYCL_TARGET=cri -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 -Ccmake.define.SGL_BUILD_MEM_FILE_LIMIT_GIB=8 -Cbuild-dir=build
  python -m pip install --force-reinstall --no-deps "$1"/wheels/*.whl
  sha256sum "$1"/wheels/*.whl
' bash "$artifact"

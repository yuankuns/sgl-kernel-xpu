#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/moe-permutation
build="$artifact/full-default-build"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=2
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_MLA_SPARSE_FUSED USE_SYCL_JIT
mkdir -p "$build"
exec 8>"$build/.lane-build.lock"
flock 8
cmake -S /root/worktrees/sgl-kernel-xpu-up/pr/muse-moe-permutation-xe35 -B "$build" -G Ninja \
  -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 \
  -DDPCPP_SYCL_TARGET=cri -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python \
  -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.9 \
  -DSKBUILD_SOABI=abi3 -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j2
cmake --install "$build" --prefix "$artifact/port/site"
cp -a /root/worktrees/sgl-kernel-xpu-up/pr/muse-moe-permutation-xe35/python/sgl_kernel/. "$artifact/port/site/sgl_kernel/"
find "$artifact/port/site" -type f -exec sha256sum {} + > "$artifact/port-site.sha256"

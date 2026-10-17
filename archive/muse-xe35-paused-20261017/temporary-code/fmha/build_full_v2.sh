#!/usr/bin/env bash
set -euo pipefail
root=/root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35
art=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export MAX_JOBS=1
exec 9>"$root/build-xe35.lock"
flock -n 9
printf 'full v2 outer Ninja jobs=8 inner MAX_JOBS=1 cpus='; taskset -pc $$
cmake -S "$root" -B "$root/build-xe35" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=cri \
  -DSKBUILD_SABI_VERSION=3.10 -DSKBUILD_SABI_COMPONENT=Development.SABIModule \
  -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python \
  -DTorch_DIR=/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DUSE_MOE=ON -DUSE_FMHA=ON -DUSE_GDN=ON -DUSE_MLA=ON -DUSE_QSA=ON \
  -DUSE_MLA_SPARSE_FUSED=OFF -DUSE_SYCL_JIT=OFF \
  -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56
cmake --build "$root/build-xe35" --parallel 8
cmake --install "$root/build-xe35" --prefix "$art/site-port"
cp -r "$root/python/sgl_kernel/." "$art/site-port/sgl_kernel/"
sha256sum "$art/site-port/sgl_kernel/"*.so > "$art/port-binary-sha256.txt"

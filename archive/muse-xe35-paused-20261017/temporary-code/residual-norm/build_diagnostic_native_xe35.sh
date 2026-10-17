#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$artifact/diagnostic-cache"
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=1
test -f "$artifact/full_build_paused_pids.json"
exec 9>"$artifact/.diagnostic-build.lock"
flock 9
cmake -S "$artifact/diagnostic-native" -B "$artifact/diagnostic-native-build" -G Ninja \
  -DPython3_EXECUTABLE="$tree/.venv-review/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build "$artifact/diagnostic-native-build" -j2
cp -a "$artifact/source-full-venv/." "$artifact/port-diagnostic-venv/"
site="$artifact/port-diagnostic-venv/lib/python3.12/site-packages"
cmake --install "$artifact/diagnostic-native-build" --prefix "$site"
cd "$artifact"
"$artifact/port-diagnostic-venv/bin/python" -c 'import sgl_kernel, torch; print("package", sgl_kernel.__file__); print("common_ops", sgl_kernel.common_ops.__file__); print("original_op", torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm)'
sha256sum "$site/sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so" "$site/sgl_kernel/libsgl-ops-sycl-FusedNormResidual.so"
nm -D -C "$site/sgl_kernel/libsgl-ops-sycl-FusedNormResidual.so" | grep 'fused_post_norm_residual_pre_norm'

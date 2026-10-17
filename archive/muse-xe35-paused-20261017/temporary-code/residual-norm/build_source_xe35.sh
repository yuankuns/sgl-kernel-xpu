#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1
export PIP_CACHE_DIR="$tree/.pip-cache-review" XDG_CACHE_HOME="$tree/.cache-review"
exec 9>"$artifact/.source-build.lock"
flock 9
cmake -S "$artifact/source-harness" -B "$artifact/source-build" -G Ninja \
  -DPython3_EXECUTABLE="$tree/.venv-review/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build "$artifact/source-build" -j1
cmake --install "$artifact/source-build" --prefix "$artifact/source-package"
cd "$artifact/source-package"
"$tree/.venv-review/bin/python" -m pip wheel -v --no-build-isolation --no-deps \
  --wheel-dir "$artifact/source-wheels" .
"$tree/.venv-review/bin/python" -m pip install --no-deps "$artifact"/source-wheels/*.whl
cd "$artifact"
"$tree/.venv-review/bin/python" -c 'import muse_residual_norm_source_xe35, torch; print(muse_residual_norm_source_xe35.__file__); print(torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm)'
sha256sum "$artifact"/source-wheels/*.whl "$artifact/source-build/libsgl-ops-sycl-FusedNormResidual.so"

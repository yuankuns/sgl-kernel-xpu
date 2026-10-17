#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
cd "$tree"
exec 9>"$tree/.build-review.lock"
flock 9
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_MLA_SPARSE_FUSED USE_SYCL_JIT
export CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=1
export PIP_CACHE_DIR="$tree/.pip-cache-review"
export XDG_CACHE_HOME="$tree/.cache-review"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
if [[ ! -f "$tree/.venv-review/.dependencies-ready" ]]; then
  cp -a /opt/gfx-deps/venv/. "$tree/.venv-review/"
  rm -rf "$tree/.venv-review/lib/python3.12/site-packages/sgl_kernel" \
    "$tree/.venv-review/lib/python3.12/site-packages/sgl_kernel-0.11.0.dist-info"
  touch "$tree/.venv-review/.dependencies-ready"
fi
"$tree/.venv-review/bin/python" -m pip wheel -v --no-deps --no-build-isolation \
  -Ccmake.define.DPCPP_SYCL_TARGET=cri -Cbuild.tool-args=-j8 \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 \
  --wheel-dir "$artifact/port-wheels" .
"$tree/.venv-review/bin/python" -m pip install --no-deps "$artifact"/port-wheels/*.whl
cd "$artifact"
"$tree/.venv-review/bin/python" -c 'import sgl_kernel, torch; print("package", sgl_kernel.__file__); print("torch", torch.__version__); print("op", torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm_xe35)'
sha256sum "$artifact"/port-wheels/*.whl "$tree"/build/src/libsgl-ops-sycl-FusedNormResidual.so

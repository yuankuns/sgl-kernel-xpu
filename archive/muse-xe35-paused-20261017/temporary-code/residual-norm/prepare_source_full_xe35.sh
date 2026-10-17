#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
source_artifact=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/residual-final-full-release
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$tree/.cache-review"
test "$(cat "$source_artifact/full-build.exit")" = 0
test "$(sha256sum "$source_artifact/frozen-source/src/sycl/FusedNormResidual.cpp" | cut -d' ' -f1)" = f8195d347b32f88da0c0203eb8082f6a3f1767b7173b789b18522b8d47c8785a
test "$(sha256sum "$artifact/source-snapshot/src/sycl/FusedNormResidual.cpp" | cut -d' ' -f1)" = f8195d347b32f88da0c0203eb8082f6a3f1767b7173b789b18522b8d47c8785a
sha256sum -c "$source_artifact/full-built-sha256.txt"
cp -a "$tree/.venv-review/." "$artifact/source-full-venv/"
site="$artifact/source-full-venv/lib/python3.12/site-packages"
rm -rf "$site/sgl_kernel" "$site/muse_residual_norm_source_xe35" \
  "$site/muse_residual_norm_source_xe35-0.0.1.dist-info"
cp -a "$source_artifact/stage/sgl_kernel" "$site/sgl_kernel"
cp "$source_artifact/source-freeze.json" "$artifact/source-full-freeze.json"
cp "$source_artifact/full-built-sha256.txt" "$artifact/source-full-original-binaries.sha256"
cp "$source_artifact/complete-package-proof.json" "$artifact/source-full-package-proof.json"
cp "$source_artifact/full-build-completion-proof.json" "$artifact/source-full-build-proof.json"
cp "$source_artifact/compile-source-proof.json" "$artifact/source-full-compile-proof.json"
cd "$artifact"
"$artifact/source-full-venv/bin/python" -c 'import sgl_kernel, torch, sys; print("prefix", sys.prefix); print("package", sgl_kernel.__file__); print("common_ops", sgl_kernel.common_ops.__file__); print("op", torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm)'
find "$site/sgl_kernel" -type f -name '*.so' -exec sha256sum {} + > "$artifact/source-full-isolated-binaries.sha256"

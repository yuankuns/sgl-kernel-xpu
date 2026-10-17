#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
target=/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35
exec 9>"$artifact/target-build.lock"
flock 9
unset USE_MOE USE_FMHA USE_GDN USE_MLA USE_QSA USE_MLA_SPARSE_FUSED USE_SYCL_JIT CMAKE_ARGS SKBUILD_CMAKE_ARGS
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
if [[ ! -f "$artifact/target-venv/dependencies-ready" ]]; then
  python - "$artifact/target-venv" <<'PY'
import pathlib, shutil, sys
destination = pathlib.Path(sys.argv[1])
if destination.exists():
    shutil.rmtree(destination)
shutil.copytree('/opt/gfx-deps/venv', destination, symlinks=True,
                ignore=shutil.ignore_patterns('sgl_kernel', 'sgl_kernel-*.dist-info', 'sglang_kernel_xpu-*.dist-info'))
(destination / 'dependencies-ready').touch()
PY
fi
cd "$target"
"$artifact/target-venv/bin/python" -m pip wheel --no-build-isolation --no-deps --no-cache-dir -v . \
  -Cbuild-dir="$artifact/target-default-build" \
  -Ccmake.define.DPCPP_SYCL_TARGET=cri \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/deps/sycl-tla-7b56 \
  -w "$artifact/target-wheels"
"$artifact/target-venv/bin/python" -m pip install --no-deps --no-cache-dir --force-reinstall "$artifact"/target-wheels/*.whl
cd /root
"$artifact/target-venv/bin/python" -c 'import sgl_kernel,torch; print(sgl_kernel.__file__); print(torch.ops.sgl_kernel.nvfp4_woq_bf16_grouped_mm_xe35)'
sha256sum "$artifact"/target-wheels/*.whl

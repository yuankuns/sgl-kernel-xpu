#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
exec 9>"$artifact/source-bridge-build.lock"
flock 9
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
cd "$artifact/baseline-package"
python -m pip wheel --no-build-isolation --no-deps --no-cache-dir -v . -w "$artifact/baseline-wheels"
sha256sum "$artifact"/baseline-wheels/*.whl

#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export PYTHONDONTWRITEBYTECODE=1
exec /opt/gfx-deps/venv/bin/python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/build_matched_baselines.py --resource-retry-v4

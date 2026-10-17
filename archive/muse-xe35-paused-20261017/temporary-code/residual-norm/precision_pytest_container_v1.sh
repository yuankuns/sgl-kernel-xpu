#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
output="$artifact/normal-precision-pytest-v1"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$output/cache" TMPDIR="$output/tmp"
mkdir -p "$TMPDIR"
[[ "$#" == 1 && ( "$1" == preflight || "$1" == runtime ) ]] || exit 2
python="$artifact/source-full-venv/bin/python"
source_python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
stage() {
  "$python" -B "$artifact/precision_pytest_stage_v1.py" --phase "$1" --python "$2" --mode "$3"
}
unset PYTHONPATH
if [[ "$1" == preflight ]]; then
  export ONEAPI_DEVICE_SELECTOR=level_zero:cpu
  stage CPU-SOURCE "$source_python" CPU-SOURCE
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  stage CPU-AFTER "$python" CPU-AFTER
  stage CPU-binding "$python" binding
else
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  status_all=0
  stage SOURCE "$source_python" SOURCE || status_all=$?
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  if [[ "$status_all" == 0 ]]; then stage AFTER "$python" AFTER || status_all=$?; fi
  stage post-protected-source "$python" post || status_all=1
  exit "$status_all"
fi

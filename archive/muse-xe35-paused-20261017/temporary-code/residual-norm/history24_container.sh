#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
output="$artifact/history24-accuracy-v1"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$output/cache" TMPDIR="$output/tmp"
mkdir -p "$TMPDIR"
[[ "$#" == 1 && ( "$1" == preflight || "$1" == runtime ) ]] || exit 2
python="$artifact/source-full-venv/bin/python"
source_python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
stage() {
  local name="$1" interpreter="$2" mode="$3"
  "$python" -B "$artifact/history24_stage.py" --phase "$name" --python "$interpreter" --mode "$mode"
}
unset PYTHONPATH
if [[ "$1" == preflight ]]; then
  stage CPU-smoke "$python" CPU-smoke
  stage CPU-SOURCE "$source_python" CPU-SOURCE
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  stage CPU-AFTER "$python" CPU-AFTER
  stage CPU-binding "$python" binding
else
  status_all=0
  stage generator "$source_python" generator || status_all=$?
  if [[ "$status_all" == 0 ]]; then stage fixture-gate "$python" fixture-gate || status_all=$?; fi
  if [[ "$status_all" == 0 ]]; then stage SOURCE "$source_python" SOURCE || status_all=$?; fi
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  if [[ "$status_all" == 0 ]]; then stage AFTER "$python" AFTER || status_all=$?; fi
  if [[ "$status_all" == 0 ]]; then stage full-comparison "$python" compare || status_all=$?; fi
  if [[ "$status_all" == 0 ]]; then stage historical17 "$python" historical || status_all=$?; fi
  stage post-protected-source "$python" post || status_all=1
  exit "$status_all"
fi

#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ "$#" == 1 && ( "$1" == cpu || "$1" == runtime ) ]] || exit 2
python="$artifact/source-full-venv/bin/python"
source_python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
runner="$artifact/residual_normal_full_perf.py"
output="$artifact/normal-full-perf-gpu4-v1"
export XDG_CACHE_HOME="$output/cache" TMPDIR="$output/tmp"
phase() {
  local name="$1" interpreter="$2"
  shift 2
  "$python" -B "$artifact/residual_normal_full_perf_stage.py" --phase "$name" --python "$interpreter" -- "$@"
}
if [[ "$1" == cpu ]]; then
  export ONEAPI_DEVICE_SELECTOR=level_zero:cpu
  mkdir "$output"
  mkdir "$TMPDIR"
  unset PYTHONPATH
  phase CPU-dataset "$python" --mode cpu
  phase CPU-SOURCE "$source_python" --mode provider --role SOURCE
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  phase CPU-AFTER "$python" --mode provider --role AFTER
else
  [[ "${ZE_AFFINITY_MASK:-}" == 4 ]] || exit 2
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  for round in 1 2 3; do
    for slot in 0 1 2 3; do
      role=AFTER
      interpreter="$python"
      export PYTHONPATH="$artifact/canonical-v4-icpx/site"
      if [[ "$slot" == 1 || "$slot" == 2 ]]; then
        role=SOURCE
        interpreter="$source_python"
        unset PYTHONPATH
      fi
      phase "round$round-slot$slot-$role" "$interpreter" --mode run --role "$role" --round "$round" --slot "$slot"
    done
  done
fi

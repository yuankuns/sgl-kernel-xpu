#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
test "$#" = 1 || exit 2
mode="$1"
test "$mode" = preflight || test "$mode" = runtime || exit 2
run_phase() {
  local name="$1"
  shift
  set +e
  "$@" > "$artifact/source-compiler-accuracy-v1/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$artifact/source-compiler-accuracy-v1/$name.exit"
  tail -n 2 "$artifact/source-compiler-accuracy-v1/$name.log"
  return "$status"
}
for provider in historical fresh; do
  python="$artifact/source-full-venv/bin/python"
  if [[ "$provider" == fresh ]]; then
    python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
  fi
  if [[ "$mode" == preflight ]]; then
    run_phase "${provider}-CPU-torch" "$python" "$artifact/probe_torch_cpu.py"
    run_phase "${provider}-CPU-normal-import" "$python" "$artifact/source_compiler_accuracy.py" --mode import --provider "$provider"
  else
    run_phase "${provider}-accuracy" "$python" "$artifact/source_compiler_accuracy.py" --mode run --provider "$provider"
  fi
done
if [[ "$mode" == runtime ]]; then
  run_phase fulltensor-comparison "$python" "$artifact/source_compiler_accuracy.py" --mode compare
fi

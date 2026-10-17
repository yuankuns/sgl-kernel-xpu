#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
output="$artifact/after-pre-wave-accuracy-v4"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$artifact/canonical-v4-icpx/site"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$output/cache"
export TMPDIR="$output/tmp"
mkdir -p "$TMPDIR"
python="$artifact/source-full-venv/bin/python"
[[ "$#" == 1 && ( "$1" == preflight || "$1" == runtime ) ]] || exit 2
run_phase() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$output/$name.command"
  printf '\n' >> "$output/$name.command"
  set +e
  "$@" > "$output/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$output/$name.exit"
  tail -n 2 "$output/$name.log"
  return "$status"
}
if [[ "$1" == preflight ]]; then
  run_phase CPU-Torch "$python" -B "$artifact/probe_torch_cpu.py"
  run_phase CPU-provider-protected-source "$python" -B "$artifact/after_pre_wave_accuracy_v4.py" --mode preflight
else
  run_phase AFTER-accuracy "$python" -B "$artifact/after_pre_wave_accuracy_v4.py" --mode accuracy
  run_phase SOURCE-AFTER-fulltensor "$python" -B "$artifact/after_pre_wave_accuracy_v4.py" --mode compare
  run_phase original-Xe35-precision-pytest "$python" -B -m pytest -q "$artifact/../../pr/muse-residual-norm-xe35/tests/test_norm.py" \
    -k xe35 -o "cache_dir=$output/pytest-cache" --junitxml="$output/precision-tests.xml"
  run_phase post-protected-source "$python" -B "$artifact/after_pre_wave_accuracy_v4.py" --mode post
fi

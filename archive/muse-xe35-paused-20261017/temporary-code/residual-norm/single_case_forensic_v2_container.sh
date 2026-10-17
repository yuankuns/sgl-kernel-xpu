#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
output="$artifact/single-case-forensic-v2"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$output/cache" TMPDIR="$output/tmp"
mkdir -p "$TMPDIR"
[[ "$#" == 1 && ( "$1" == preflight || "$1" == runtime ) ]] || exit 2
python="$artifact/source-full-venv/bin/python"
source_python="$artifact/source-before-icpx-bounded-v1/isolated-source-venv/bin/python"
run_phase() {
  local name="$1"
  shift
  printf "%q " "$@" > "$output/$name.command"
  printf "\n" >> "$output/$name.command"
  "$@" > "$output/$name.log" 2>&1 &
  local child=$!
  echo "$child" > "$output/$name.pid"
  set +e
  wait "$child"
  local status=$?
  set -e
  echo "$status" > "$output/$name.exit"
  tail -n 3 "$output/$name.log"
  return "$status"
}
unset PYTHONPATH
if [[ "$1" == preflight ]]; then
  run_phase CPU-Torch "$python" -B "$artifact/probe_torch_cpu.py"
  run_phase CPU-data "$python" -B "$artifact/single_case_forensic_v2.py" --mode CPU-data
  run_phase CPU-SOURCE "$source_python" -B "$artifact/single_case_forensic_v2.py" --mode CPU-SOURCE
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  run_phase CPU-AFTER "$python" -B "$artifact/single_case_forensic_v2.py" --mode CPU-AFTER
  run_phase CPU-binding "$python" -B "$artifact/single_case_forensic_v2.py" --mode binding
else
  status_all=0
  run_phase SOURCE "$source_python" -B "$artifact/single_case_forensic_v2.py" --mode SOURCE || status_all=$?
  export PYTHONPATH="$artifact/canonical-v4-icpx/site"
  if [[ "$status_all" == 0 ]]; then
    run_phase AFTER "$python" -B "$artifact/single_case_forensic_v2.py" --mode AFTER || status_all=$?
  fi
  if [[ "$status_all" == 0 ]]; then
    run_phase full-comparison "$python" -B "$artifact/single_case_forensic_v2.py" --mode compare || status_all=$?
  fi
  run_phase post-protected-source "$python" -B "$artifact/single_case_forensic_v2.py" --mode post || status_all=1
  exit "$status_all"
fi

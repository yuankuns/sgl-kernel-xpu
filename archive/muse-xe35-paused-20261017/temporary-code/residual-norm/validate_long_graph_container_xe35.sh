#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$tree/.cache-review"
source_python="$artifact/source-full-venv/bin/python"
port_python="$artifact/port-diagnostic-venv/bin/python"
bench="$tree/benchmark/bench_fused_norm_residual.py"
cd "$artifact"
run_long_graph() {
  local name="$1"
  shift
  set +e
  "$@" > "$artifact/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$artifact/$name.exit"
  tail -n 2 "$artifact/$name.log"
  return "$status"
}
for environment in baseline reference_default reference_kmp0; do
  environment_args=(env -u KMP_BLOCKTIME)
  check_args=()
  if [[ "$environment" != baseline ]]; then
    check_args=(--check)
  fi
  if [[ "$environment" == reference_kmp0 ]]; then
    environment_args=(env KMP_BLOCKTIME=0)
  fi
  for calls in 128 256; do
    for round in 1 2; do
      for pair in B1:port-native A1:source A2:source B2:port-native; do
        name="${pair%%:*}"
        provider="${pair##*:}"
        provider_python="$source_python"
        if [[ "$provider" == port-native ]]; then
          provider_python="$port_python"
        fi
        prefix="long_graph_${environment}_${calls}_round${round}_"
        run_long_graph "${prefix}perf_$name" "${environment_args[@]}" "$provider_python" "$bench" \
          --provider "$provider" --decode-focus --cpu-submit-trace --iterations "$calls" --samples 31 "${check_args[@]}"
      done
    done
  done
done
run_long_graph long_graph_analysis "$port_python" "$artifact/analyze_long_graph.py"

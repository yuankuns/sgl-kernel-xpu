#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$tree/.cache-review"
bench="$tree/benchmark/bench_fused_norm_residual.py"
source_python="$artifact/source-full-venv/bin/python"
port_python="$artifact/port-diagnostic-venv/bin/python"
cd "$artifact"
run_xe35() {
  local name="$1"
  shift
  set +e
  "$@" >"$artifact/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" >"$artifact/$name.exit"
  tail -n 3 "$artifact/$name.log"
  return "$status"
}
if [[ ! -f "$artifact/source_accuracy.exit" ]] || [[ "$(cat "$artifact/source_accuracy.exit")" != 0 ]]; then
  run_xe35 source_accuracy "$source_python" "$bench" --provider source --accuracy-only --output-dir "$artifact/source-outputs"
fi
run_xe35 diagnostic_port_accuracy "$port_python" "$bench" --provider port-native --accuracy-only --output-dir "$artifact/diagnostic-port-outputs"
run_xe35 diagnostic_compare_accuracy "$port_python" "$artifact/compare_accuracy.py" --port-log diagnostic_port_accuracy.log --port-outputs diagnostic-port-outputs
torch_sync_cmd='import torch,time; torch.xpu.synchronize(); time.sleep(5)'
"$port_python" -c "$torch_sync_cmd"
for pair in B1:port-native A1:source A2:source B2:port-native; do
  name="${pair%%:*}"
  provider="${pair##*:}"
  provider_python="$source_python"
  if [[ "$provider" == port-native ]]; then
    provider_python="$port_python"
  fi
  run_xe35 "diagnostic_perf_$name" "$provider_python" "$bench" --provider "$provider"
done
run_xe35 diagnostic_perf_summary "$port_python" "$artifact/summarize_perf.py" --prefix diagnostic_
if "$port_python" -c 'import json,sys; sys.exit(0 if any(case["retest_required"] for case in json.load(open("diagnostic_perf_summary.json"))) else 1)'; then
  printf '%s\n' '>5% diagnostic regression found; running a separate matched GPU7 BAAB retest, no automatic performance acceptance.'
  for pair in B1:port-native A1:source A2:source B2:port-native; do
    name="${pair%%:*}"
    provider="${pair##*:}"
    provider_python="$source_python"
    if [[ "$provider" == port-native ]]; then
      provider_python="$port_python"
    fi
    run_xe35 "diagnostic_retest_perf_$name" "$provider_python" "$bench" --provider "$provider"
  done
  run_xe35 diagnostic_retest_perf_summary "$port_python" "$artifact/summarize_perf.py" --prefix diagnostic_retest_
fi
run_xe35 diagnostic_perf_review "$port_python" "$artifact/review_performance.py" --prefix diagnostic_

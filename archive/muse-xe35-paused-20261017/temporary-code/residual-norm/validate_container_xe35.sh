#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XDG_CACHE_HOME="$tree/.cache-review"
python="$tree/.venv-review/bin/python"
source_python="$artifact/source-full-venv/bin/python"
bench="$tree/benchmark/bench_fused_norm_residual.py"
cd "$artifact"
run_xe35() {
  local name="$1"
  shift
  set +e
  "$@" >"$artifact/$name.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$artifact/$name.exit"
  tail -n 4 "$artifact/$name.log"
  return "$status"
}
run_xe35 gpu_info "$python" -c 'import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_properties(0)); print(torch.xpu.device_count())'
if [[ ! -f "$artifact/source_accuracy.exit" ]] || [[ "$(cat "$artifact/source_accuracy.exit")" != 0 ]]; then
  run_xe35 source_accuracy "$source_python" "$bench" --provider source --accuracy-only --output-dir "$artifact/source-outputs"
fi
run_xe35 port_accuracy "$python" "$bench" --provider port --accuracy-only --output-dir "$artifact/port-outputs"
run_xe35 compare_accuracy "$python" "$artifact/compare_accuracy.py"
run_xe35 port_pytest "$python" -m pytest -q "$tree/tests/test_norm.py" -k xe35
for pair in B1:port A1:source A2:source B2:port; do
  name="${pair%%:*}"
  provider="${pair##*:}"
  provider_python="$python"
  if [[ "$provider" == source ]]; then
    provider_python="$source_python"
  fi
  run_xe35 "perf_$name" "$provider_python" "$bench" --provider "$provider"
done
run_xe35 perf_summary "$python" "$artifact/summarize_perf.py"
if "$python" -c 'import json,sys; sys.exit(0 if any(case["retest_required"] for case in json.load(open("perf_summary.json"))) else 1)'; then
  printf '%s\n' '>5% regression found; running a separate matched GPU7 BAAB retest, no automatic performance acceptance.'
  for pair in B1:port A1:source A2:source B2:port; do
    name="${pair%%:*}"
    provider="${pair##*:}"
    provider_python="$python"
    if [[ "$provider" == source ]]; then
      provider_python="$source_python"
    fi
    run_xe35 "retest_perf_$name" "$provider_python" "$bench" --provider "$provider"
  done
  run_xe35 retest_perf_summary "$python" "$artifact/summarize_perf.py" --prefix retest_
fi
run_xe35 perf_review "$python" "$artifact/review_performance.py"

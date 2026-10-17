#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for variable in BATCHES BATCH_SIZE SERVING_BATCH; do
  if [[ "${!variable-8}" != 8 ]]; then
    printf '%s\n' 'Only B8 / ISL4096 / OSL16 / TP1 / EP1 is accepted' >&2
    exit 2
  fi
done
for component in memory position; do
  artifact="$root/$component"
  for run in B0 A0 A1 B1; do
    case "$run" in B*) label=before ;; A*) label=after ;; esac
    output="$artifact/diagnostic-repeat1-$run"
    if [[ -e "$output.json" || -e "$output.exit" ]]; then exit 2; fi
    set +e
    PYTHONPATH="$artifact/diagnostic/$label/site" python "$root/benchmark_simple.py" \
      "$component" --api source-native --label "$label" --samples 101 --calls 128 \
      --output "$output.json" > "$output.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$output.exit"
    if (( status != 0 )); then tail -c 5000 "$output.log"; exit "$status"; fi
    sleep 2
  done
  python "$root/analyze_simple_diagnostic.py" "$component" --prefix diagnostic-repeat1 \
    > "$artifact/diagnostic-repeat1-analysis.log" 2>&1
done

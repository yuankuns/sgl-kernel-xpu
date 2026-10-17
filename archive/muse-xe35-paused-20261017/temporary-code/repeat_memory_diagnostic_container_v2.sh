#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
artifact="$root/memory"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for variable in BATCHES BATCH_SIZE SERVING_BATCH; do
  if [[ "${!variable-8}" != 8 ]]; then exit 2; fi
done
for round in 0 1 2; do
  for run in B0 A0 A1 B1; do
    case "$run" in B*) label=before ;; A*) label=after ;; esac
    output="$artifact/diagnostic-repeat2-r$round-$run"
    if [[ -e "$output.json" || -e "$output.exit" ]]; then exit 2; fi
    set +e
    PYTHONPATH="$artifact/diagnostic/$label/site" python "$root/benchmark_simple.py" \
      memory --api source-native --label "$label" --samples 101 --calls 128 \
      --output "$output.json" > "$output.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$output.exit"
    if (( status != 0 )); then tail -c 5000 "$output.log"; exit "$status"; fi
    sleep 2
  done
  python "$root/analyze_simple_diagnostic.py" memory --prefix "diagnostic-repeat2-r$round" \
    > "$artifact/diagnostic-repeat2-r$round-analysis.log" 2>&1
done

#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for variable in BATCHES BATCH_SIZE SERVING_BATCH; do
  if [[ "${!variable-8}" != 8 ]]; then
    printf '%s\n' 'Only Muse-Spark B8 / ISL4096 / OSL16 / TP1 / EP1 is accepted' >&2
    exit 2
  fi
done
for component in "$@"; do
  case "$component" in memory|position|logits-vocab) ;; *) exit 2 ;; esac
  artifact="$root/$component"
  cd "$artifact/native-diagnostic"
  for label in before after; do
    set +e
    PYTHONPATH="$artifact/diagnostic/$label/site:$root" \
      python -m pytest -p acceptance_cooldown tests -q \
      > "$artifact/diagnostic-accuracy-$label.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$artifact/diagnostic-accuracy-$label.exit"
    if (( status != 0 )); then tail -c 5000 "$artifact/diagnostic-accuracy-$label.log"; exit "$status"; fi
  done
  for run in B0 A0 A1 B1; do
    case "$run" in B*) label=before ;; A*) label=after ;; esac
    set +e
    PYTHONPATH="$artifact/diagnostic/$label/site" python "$root/benchmark_simple.py" \
      "$component" --api source-native --label "$label" \
      --output "$artifact/diagnostic-perf-$run.json" \
      > "$artifact/diagnostic-perf-$run.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$artifact/diagnostic-perf-$run.exit"
    if (( status != 0 )); then tail -c 5000 "$artifact/diagnostic-perf-$run.log"; exit "$status"; fi
    sleep 2
  done
done

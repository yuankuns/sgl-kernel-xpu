#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for variable in BATCHES BATCH_SIZE SERVING_BATCH; do
  if [[ "${!variable-8}" != 8 ]]; then exit 2; fi
done
if (( $# == 0 )); then set -- memory position logits-vocab; fi
for component in "$@"; do
  case "$component" in memory|position|logits-vocab) ;; *) exit 2 ;; esac
  artifact="$root/$component"
  worktree="/root/worktrees/sgl-kernel-xpu-up/pr/muse-$component-xe35"
  python "$root/stage_simple_official_baseline_v2.py" "$component" \
    > "$artifact/official-baseline-stage-v2.log" 2>&1
  for label in before after; do
    PYTHONPATH="$artifact/$label/site" python "$root/check_simple_official_package.py" "$component" "$label" \
      > "$artifact/official-package-$label-preflight.json" 2> "$artifact/official-package-$label-preflight.log"
  done
  case "$component" in
    memory) tests=(tests/test_paged_alloc.py tests/test_write_req_to_token_pool.py) ;;
    position) tests=(tests/test_compute_position.py) ;;
    logits-vocab) tests=(tests/test_softcap_logits.py tests/test_vocab_mask.py) ;;
  esac
  cd "$worktree"
  set +e
  PYTHONPATH="$artifact/after/site:$root/logits-vocab/framework-original/site:$root" python -m pytest -p acceptance_cooldown "${tests[@]}" -q \
    > "$artifact/port-accuracy.log" 2>&1
  test_status=$?
  set -e
  printf '%s\n' "$test_status" > "$artifact/port-accuracy.exit"
  if (( test_status != 0 )); then tail -c 5000 "$artifact/port-accuracy.log"; exit "$test_status"; fi
  for run in B0 A0 A1 B1; do
    case "$run" in B*) label=before ;; A*) label=after ;; esac
    set +e
    PYTHONPATH="$artifact/$label/site" python "$root/benchmark_simple.py" \
      "$component" --label "$label" --samples 101 --calls 128 --output "$artifact/perf-$run.json" \
      > "$artifact/perf-$run.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$artifact/perf-$run.exit"
    if (( status != 0 )); then tail -c 5000 "$artifact/perf-$run.log"; exit "$status"; fi
    sleep 2
  done
done

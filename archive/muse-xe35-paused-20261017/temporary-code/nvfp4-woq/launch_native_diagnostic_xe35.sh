#!/usr/bin/env bash
set -euo pipefail
for specification in BATCHES:8 BATCH:8 BATCH_SIZE:8 SERVING_BATCH:8 ISL:4096 OSL:16 TP:1 EP:1; do
  name=${specification%:*}
  expected=${specification#*:}
  if [[ -v "$name" && "${!name}" != "$expected" ]]; then
    printf '%s must be %s before GPU work\n' "$name" "$expected" >&2
    exit 2
  fi
done
run="${1:-firstpass}"
mode="${2:-all}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
if [[ "$mode" != tests && "$mode" != tails ]]; then
  printf '%s\n' 'GPU9 is accuracy-only: performance/all/kernels disabled; NV performance requires main GPU5 grant after FMHA' >&2
  exit 2
fi
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
test -f "$artifact/diagnostic.READY"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 9 --workdir /root \
  -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/run_native_diagnostic_container_xe35.sh "$run" "$mode" \
  >"$artifact/$run-native-diagnostic.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/$run-native-diagnostic.exit"
tail -n 30 "$artifact/$run-native-diagnostic.log"
exit "$status"

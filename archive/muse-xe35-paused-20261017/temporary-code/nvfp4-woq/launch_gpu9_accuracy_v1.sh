#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
for specification in BATCHES:8 BATCH:8 BATCH_SIZE:8 SERVING_BATCH:8 ISL:4096 OSL:16 TP:1 EP:1; do
  name=${specification%:*}
  expected=${specification#*:}
  if [[ -v "$name" && "${!name}" != "$expected" ]]; then
    printf '%s must be %s before GPU work\n' "$name" "$expected" >&2
    exit 2
  fi
done
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 9 --workdir /root -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/run_gpu9_accuracy_v1.sh \
  >"$artifact/grf512-accuracy-only-native.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/grf512-accuracy-only-native.exit"
tail -n 25 "$artifact/grf512-accuracy-only-native.log"
exit "$status"

#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
run_name=${RUN_NAME:-target-default-build-offline}
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
"$wrapper" --workdir /root -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/build_target_xe35.sh \
  >"$artifact/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/$run_name.exit"
tail -n 50 "$artifact/$run_name.log"
exit "$status"

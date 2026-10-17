#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
printf '%s\n' "$BASHPID" >"$artifact/grf512-accuracy-only.supervisor.pid"
printf '%s\n' 'GPU9 accuracy-only: wait existing last timing child/true status; no performance successor'
while [[ ! -f "$artifact/diagnostic-barrier-round.exit" ]]; do sleep 5; done
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir /root -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/stage_grf512_recovery.py \
  >"$artifact/grf512-accuracy-only-stage.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/grf512-accuracy-only-stage.exit"
if [[ "$status" != 0 ]]; then exit "$status"; fi
bash "$artifact/launch_gpu9_accuracy_v1.sh"
status=$?
printf '%s\n' "$status" >"$artifact/grf512-accuracy-only-supervisor.exit"
exit "$status"

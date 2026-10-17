#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
run=grf512-repeat
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
printf '%s\n' "$BASHPID" >"$artifact/$run.supervisor.pid"
printf '%s\n' 'CPU-only recovery waiter; original shared production round must finish naturally'
while [[ ! -f "$artifact/diagnostic-barrier-round.exit" ]]; do sleep 5; done
"$wrapper" --workdir /root -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/stage_grf512_recovery.py \
  >"$artifact/$run-stage.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/$run-stage.exit"
if [[ "$status" != 0 ]]; then
  cat "$artifact/$run-stage.log"
  exit "$status"
fi
bash "$artifact/launch_native_diagnostic_xe35.sh" "$run" all
exit "$?"

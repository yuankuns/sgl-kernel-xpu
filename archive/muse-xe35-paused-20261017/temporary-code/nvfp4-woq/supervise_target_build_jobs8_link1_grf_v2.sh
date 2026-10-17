#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
run_name=target-default-build-jobs8-link1-grf-v2
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
printf '%s\n' "$BASHPID" >"$artifact/$run_name.supervisor.pid"
date -Is >"$artifact/$run_name.started"
cat /proc/meminfo >"$artifact/$run_name.host-meminfo"
command=("$wrapper" --workdir /root -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/build_target_jobs8_link1_grf_v2.sh)
printf '%q ' "${command[@]}" >"$artifact/$run_name.command"
printf '\n' >>"$artifact/$run_name.command"
"${command[@]}" >"$artifact/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/$run_name.exit"
date -Is >"$artifact/$run_name.ended"
tail -n 35 "$artifact/$run_name.log"
exit "$status"

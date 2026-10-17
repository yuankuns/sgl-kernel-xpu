#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
job="$artifact/source-before-icpx-bounded-v1"
test "$#" = 0
test "$(cat "$artifact/source-before-icpx-prepare.exit")" = 0
exec 9>"$job/supervisor.lock"
flock -n 9
test ! -e "$job/supervisor.exit"
test ! -e "$job/runtime.json"
printf 'host_supervisor_pid=%s cpu=48-63 outer=2 MAX_JOBS=1 compilerRSS=8GiB treeRSS=16GiB GPU=false\n' "$$"
echo "$$" > "$job/host-supervisor.pid"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 3 \
  /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/source-full-venv/bin/python -B \
  /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/build_bounded_source.py \
  > "$job/supervisor.log" 2>&1
status=$?
echo "$status" > "$job/supervisor.exit"
tail -n 14 "$job/supervisor.log"
exit "$status"

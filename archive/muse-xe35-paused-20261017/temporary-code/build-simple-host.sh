#!/usr/bin/env bash
set -euo pipefail
root=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
rm -f "$root/simple-build-supervisor.exit"
printf '%s\n' 'GPU work is not performed by this compile supervisor' > "$root/simple-build-supervisor.log"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/sgl-kernel-xpu-up -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 7 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-simple-container.sh \
  >> "$root/simple-build-supervisor.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$root/simple-build-supervisor.exit"
tail -35 "$root/simple-build-supervisor.log"
exit "$status"

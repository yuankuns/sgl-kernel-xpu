#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
if [[ -f "$root/build-full.exit" ]]; then
  unlink "$root/build-full.exit"
fi
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35 -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 2 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/build-container_xe35.sh \
  > "$root/build-full.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/build-full.exit"
tail -40 "$root/build-full.log"
exit "$status"

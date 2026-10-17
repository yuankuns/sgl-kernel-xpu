#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ "$(cat "$root/build-j2-stop.exit")" == 0 ]]
[[ "$(cat "$root/build-full-j2.exit")" == 130 ]]
[[ ! -f "$root/build-full-j8-v2.exit" ]]
if [[ -f "$root/build-full.exit" ]]; then
  unlink "$root/build-full.exit"
fi
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35 -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 2 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/build-container-j8-v2_xe35.sh \
  > "$root/build-full-j8-v2.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/build-full-j8-v2.exit"
printf '%s\n' "$status" > "$root/build-full.exit"
tail -40 "$root/build-full-j8-v2.log"
exit "$status"

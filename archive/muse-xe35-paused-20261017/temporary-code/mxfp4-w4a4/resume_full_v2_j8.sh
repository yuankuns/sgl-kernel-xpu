#!/usr/bin/env bash
set -euo pipefail
artifact_root=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
[[ "$(cat "$artifact_root/mxfp4-w4a4/full-build-j2-stop-tree.exit")" == 0 ]] || exit 125
for module in mxfp4-w4a4 routing-quant; do
  artifact="$artifact_root/$module"
  target="/root/worktrees/sgl-kernel-xpu-up/pr/muse-$module-xe35"
  printf '%s\n' "SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 enter-sglang-syk.sh --workdir $target -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 4 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/build_port_full_v2_j8.sh $module" > "$artifact/full-build-v2-j8.command"
  set +e
  SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$target" -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 4 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/build_port_full_v2_j8.sh "$module" > "$artifact/full-build-v2-j8.log" 2>&1
  status=$?
  echo "$status" > "$artifact/full-build-v2-j8.exit"
  echo "$status" > "$artifact/full-build.exit"
  tail -c 2200 "$artifact/full-build-v2-j8.log"
  if [[ "$status" != 0 ]]; then exit "$status"; fi
  set -e
  echo "$module default full v2 normal package completed exit0"
done

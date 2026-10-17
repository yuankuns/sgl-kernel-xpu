#!/usr/bin/env bash
set -uo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
version=canonical-v4-icpx
if [[ "${1:-}" == --version ]]; then
  [[ "$#" -ge 2 ]] || exit 2
  version="$2"
  shift 2
fi
[[ "$version" == canonical-v4-icpx ]] || exit 2
phase="${1:-}"
run="${2:-}"
grant="${3:-}"
[[ "$#" == 2 || "$#" == 3 ]] || exit 2
[[ "$phase" == accuracy || "$phase" == performance ]] || exit 2
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
controls=(--version "$version" --phase "$phase" --run "$run")
[[ -n "${NVFP4_CANONICAL_RELEASE:-}" ]] || exit 2
controls+=(--release "$NVFP4_CANONICAL_RELEASE")
if [[ -n "$grant" ]]; then controls+=(--grant "$grant"); fi
hash=$(PYTHONDONTWRITEBYTECODE=1 python3 "$artifact/canonical_launch_preflight.py" "${controls[@]}") || exit 2
gpu=9
if [[ "$phase" == performance ]]; then
  gpu=5
  export NVFP4_GRANTED_MANIFEST_SHA256="$hash"
else
  unset NVFP4_GRANTED_MANIFEST_SHA256
fi
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu "$gpu" --workdir /root -- \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 5 \
  bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq/run_canonical_validation.sh "$phase" "$run" "$hash" "$version" \
  >"$artifact/$run-canonical.log" 2>&1
status=$?
printf '%s\n' "$status" >"$artifact/$run-canonical.exit"
tail -n 20 "$artifact/$run-canonical.log"
exit "$status"

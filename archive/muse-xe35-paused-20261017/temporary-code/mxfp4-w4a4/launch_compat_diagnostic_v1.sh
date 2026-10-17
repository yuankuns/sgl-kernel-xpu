#!/usr/bin/env bash
set -euo pipefail
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Only serving batch8; rejected before GPU.' >&2; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    *) exit 2 ;;
  esac
done
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
[[ "$(cat "$artifact/compat-package-v1.exit")" == 0 ]] || exit 125
[[ "$(cat "$artifact/fresh-diagnostic-runtime-baab.exit")" == 1 ]] || exit 125
export MUSE_BARRIER_FILE="$artifact/gpu8-compat-source-barrier-v1.json"
python3 "$artifact/wait_gpu_barrier.py" --module mxfp4-w4a4 --phase diagnostic
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 8 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4 -- taskset -c 64-79 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/run_compat_diagnostic_v1.sh > "$artifact/compat-diagnostic-runtime-baab.log" 2>&1
status=$?
echo "$status" > "$artifact/compat-diagnostic-runtime-baab.exit"
tail -c 5000 "$artifact/compat-diagnostic-runtime-baab.log"
printf '\nGPU8 W4 compatibility diagnostic released; true exit=%s\n' "$status"
exit "$status"

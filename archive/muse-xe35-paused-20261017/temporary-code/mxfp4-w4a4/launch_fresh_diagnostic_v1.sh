#!/usr/bin/env bash
set -euo pipefail
printf 'Historical fresh-source round is terminal; old GPU8 timing barrier cannot be reused.\n' >&2
exit 125
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
[[ "$(cat "$artifact/fresh-source-package.exit")" == 0 ]] || exit 125
[[ "$(cat "$artifact/../routing-quant/diagnostic-runtime-baab.exit")" =~ ^[0-9]+$ ]] || exit 125
export MUSE_BARRIER_FILE="$artifact/gpu8-fresh-source-barrier.json"
python3 "$artifact/wait_gpu_barrier.py" --module mxfp4-w4a4 --phase diagnostic
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 8 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4 -- taskset -c 64-79 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/run_fresh_diagnostic_v1.sh > "$artifact/fresh-diagnostic-runtime-baab.log" 2>&1
status=$?
echo "$status" > "$artifact/fresh-diagnostic-runtime-baab.exit"
tail -c 5000 "$artifact/fresh-diagnostic-runtime-baab.log"
printf '\nGPU8 W4 fresh-source diagnostic released; true exit=%s\n' "$status"
exit "$status"

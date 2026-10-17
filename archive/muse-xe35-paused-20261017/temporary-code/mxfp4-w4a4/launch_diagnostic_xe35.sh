#!/usr/bin/env bash
set -euo pipefail
printf 'Retired GPU8 timing launcher: GPU8 is now precision-only; use the new compat precision barrier.\n' >&2
exit 125
module="${1:?module required}"
shift
case "$module" in mxfp4-w4a4|routing-quant) ;; *) exit 2 ;; esac
phase=baab
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Only serving batch 8 is allowed; rejected before GPU work.' >&2; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    --phase) phase="${2:?phase required}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$phase" in baab|tests|all) ;; *) exit 2 ;; esac
base=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
artifact="$base/$module"
[[ -f "$artifact/diagnostic-package.exit" && "$(cat "$artifact/diagnostic-package.exit")" == 0 ]] || { echo "$module NOT READY" >&2; exit 125; }
python3 "$base/mxfp4-w4a4/wait_gpu_barrier.py" --module "$module" --phase diagnostic
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 8 --workdir "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/$module" -- taskset -c 64-79 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/run_diagnostic_xe35.sh "$module" "$phase" > "$artifact/diagnostic-runtime-$phase.log" 2>&1
status=$?
echo "$status" > "$artifact/diagnostic-runtime-$phase.exit"
tail -c 5500 "$artifact/diagnostic-runtime-$phase.log"
printf '\nGPU8 %s diagnostic released; exit=%s\n' "$module" "$status"
exit "$status"

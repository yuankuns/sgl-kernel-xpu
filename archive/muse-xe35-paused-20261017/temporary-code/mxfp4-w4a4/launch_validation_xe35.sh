#!/usr/bin/env bash
set -euo pipefail
printf 'Retired GPU8 performance launcher: main must grant W4 GPU6 or routing GPU7 after canonical full acceptance.\n' >&2
exit 125
module="${1:?module required}"
shift
case "$module" in mxfp4-w4a4) runner=run_port_xe35.sh ;; routing-quant) runner=run_validation_xe35.sh ;; *) exit 2 ;; esac
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Only serving batch 8 is allowed; rejected before GPU work.' >&2; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
artifact="/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/$module"
[[ -f "$artifact/full-build.exit" && "$(cat "$artifact/full-build.exit")" == 0 ]] || { echo "$module official full package not ready" >&2; exit 125; }
python3 /home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4/wait_gpu_barrier.py --module "$module" --phase official
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 8 --workdir "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/$module" -- taskset -c 64-79 bash "$runner" > "$artifact/official-runtime.log" 2>&1
status=$?
echo "$status" > "$artifact/official-runtime.exit"
tail -c 5500 "$artifact/official-runtime.log"
printf '\nGPU8 %s official validation released; exit=%s\n' "$module" "$status"
exit "$status"

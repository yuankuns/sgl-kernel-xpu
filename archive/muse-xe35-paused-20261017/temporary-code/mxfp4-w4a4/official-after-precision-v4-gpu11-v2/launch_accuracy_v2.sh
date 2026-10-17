#!/usr/bin/env bash
set -euo pipefail
while (( $# )); do
  case "$1" in
    --serving-batch) [[ "${2:-}" == 8 ]] || { echo 'Reject non-B8 before GPU'; exit 2; }; shift 2 ;;
    --isl) [[ "${2:-}" == 4096 ]] || exit 2; shift 2 ;;
    --osl) [[ "${2:-}" == 16 ]] || exit 2; shift 2 ;;
    --tp|--ep) [[ "${2:-}" == 1 ]] || exit 2; shift 2 ;;
    *) exit 2 ;;
  esac
done
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
evidence="$artifact/official-after-precision-v4-gpu11-v2"
container_artifact="${artifact/\/home\/syk/\/root}"
[[ ! -f "$evidence/host-supervisor.pid" ]] || { echo 'Already launched'; exit 1; }
python3 -c 'import json,sys,time; grant=json.load(open(sys.argv[1])); time.sleep(max(0,grant["start_epoch"]-time.time()))' "$evidence/gpu11-official-after-accuracy-grant-v2.json"
python3 "$evidence/prepare_grant.py" --verify > "$evidence/grant-verified.log" 2>&1
echo "$BASHPID" > "$evidence/host-supervisor.pid"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 11 --workdir "$container_artifact/official-after-precision-v4-gpu11-v2" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 bash "$artifact/official-after-precision-v4-gpu11-v2/run_accuracy_v2.sh"' bash "$container_artifact" > "$evidence/gpu-runtime.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/gpu-runtime.exit"
tail -c 4000 "$evidence/gpu-runtime.log"
printf 'W4 GPU11 normal AFTER accuracy released; true exit=%s\n' "$status"
exit "$status"

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
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/official-after-accuracy-v4-gpu11-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
[[ -f "$directory/main-normal-after-accuracy-gpu11-grant-v1.json" && ! -f "$directory/host-supervisor.pid" ]] || exit 2
taskset -c 72-79 python3 -c 'import json,sys,time; grant=json.load(open(sys.argv[1])); assert grant["issuer"]=="main" and grant["release"]; time.sleep(max(0,grant["start_epoch"]-time.time()))' "$directory/main-normal-after-accuracy-gpu11-grant-v1.json"
taskset -c 72-79 python3 "$directory/verify_grant.py" > "$directory/grant-verified.log" 2>&1
echo "$BASHPID" > "$directory/host-supervisor.pid"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 11 --workdir "$container_artifact/official-after-accuracy-v4-gpu11-v1" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 72-79 bash "$artifact/official-after-accuracy-v4-gpu11-v1/run_gpu.sh"' bash "$container_artifact" > "$directory/gpu-runtime.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/gpu-runtime.exit"
tail -c 4500 "$directory/gpu-runtime.log"
exit "$status"

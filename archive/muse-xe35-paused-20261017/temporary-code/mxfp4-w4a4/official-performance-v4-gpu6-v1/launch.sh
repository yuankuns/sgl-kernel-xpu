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
directory="$artifact/official-performance-v4-gpu6-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
[[ ! -f "$directory/host-supervisor.pid" ]] || exit 1
echo "$BASHPID" > "$directory/host-supervisor.pid"
taskset -c 32-47 python3 -c 'import json,sys,time; grant=json.load(open(sys.argv[1])); time.sleep(max(0,grant["start_epoch"]-time.time()))' "$directory/main-performance-gpu6-grant-v1.json"
taskset -c 32-47 python3 "$directory/prepare_grant.py" --verify > "$directory/grant-verified.log" 2>&1
set +e
taskset -c 32-47 python3 "$directory/maintenance_snapshot.py" --output "$directory/maintenance-final-before-GPU.json" > "$directory/maintenance-final-before-GPU.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/maintenance-final-before-GPU.exit"
(( status == 0 )) || exit "$status"
taskset -c 32-47 python3 -c 'import json,sys; before=json.load(open(sys.argv[1])); after=json.load(open(sys.argv[2])); assert before["stack"]==after["stack"]' "$directory/CPU-maintenance-stack.json" "$directory/maintenance-final-before-GPU.json"
taskset -c 32-47 python3 "$directory/maintenance_snapshot.py" --watch > "$directory/maintenance-monitor.log" 2>&1 &
monitor_pid=$!
echo "$monitor_pid" > "$directory/maintenance-monitor.pid"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 6 --workdir "$container_artifact/official-performance-v4-gpu6-v1" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 32-47 bash "$artifact/official-performance-v4-gpu6-v1/run_gpu.sh"' bash "$container_artifact" > "$directory/gpu-runtime.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/gpu-runtime.exit"
touch "$directory/maintenance-monitor-stop"
wait "$monitor_pid"
if [[ -f "$directory/performance-maintenance-hold.json" ]] && (( status == 0 )); then status=125; fi
tail -c 4500 "$directory/gpu-runtime.log"
exit "$status"

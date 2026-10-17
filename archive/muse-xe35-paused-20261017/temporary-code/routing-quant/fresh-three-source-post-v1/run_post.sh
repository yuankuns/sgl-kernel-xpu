#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/fresh-three-source-post-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
echo "$BASHPID" > "$directory/host-post-supervisor.pid"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/fresh-three-source-post-v1" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 env ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site" /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind fresh --output "$artifact/fresh-three-source-post-v1/package-post-fresh.json"' bash "$container_artifact" > "$directory/package-post-fresh.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/package-post-fresh.exit"
(( status == 0 )) || exit "$status"
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/fresh-three-source-post-v1" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 env ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH="$artifact/fresh-source-icpx-native-v2/site:$artifact/../mxfp4-w4a4" /opt/gfx-deps/venv/bin/python "$artifact/fresh-three-source-post-v1/audit_fixtures.py"' bash "$container_artifact" > "$directory/CPU-full-fixture-audit.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-full-fixture-audit.exit"
(( status == 0 )) || exit "$status"
set +e
taskset -c 64-79 python3 "$directory/capture_drain.py" > "$directory/CPU-terminal-classification.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-terminal-classification.exit"
tail -c 4000 "$directory/CPU-terminal-classification.log"
exit "$status"

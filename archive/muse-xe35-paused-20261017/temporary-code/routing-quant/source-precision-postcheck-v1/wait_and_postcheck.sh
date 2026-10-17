#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/source-precision-postcheck-v1"
evidence="$artifact/historical-fresh-source-precision-v1"
container_artifact="${artifact/\/home\/syk/\/root}"
[[ ! -f "$directory/host-supervisor.pid" ]] || exit 1
echo "$BASHPID" > "$directory/host-supervisor.pid"
while [[ ! -f "$evidence/host-launch.exit" ]]; do sleep 10; done
for kind in historical fresh; do
  site="$container_artifact/fresh-source-icpx-native-v2/site"
  if [[ "$kind" == historical ]]; then site="$container_artifact/historical-source-normal-v1/site"; fi
  set +e
  SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/source-precision-postcheck-v1" -- bash -c 'artifact="$1"; site="$2"; kind="$3"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 env ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$site" /opt/gfx-deps/venv/bin/python "$artifact/check_source_precision_package.py" --kind "$kind" --output "$artifact/source-precision-postcheck-v1/package-post-$kind.json"' bash "$container_artifact" "$site" "$kind" > "$directory/package-post-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$directory/package-post-$kind.exit"
  if (( status != 0 )); then tail -c 3000 "$directory/package-post-$kind.log"; exit "$status"; fi
done
set +e
taskset -c 64-79 python3 "$directory/audit_terminal.py" > "$directory/CPU-result-audit.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/CPU-result-audit.exit"
tail -c 4000 "$directory/CPU-result-audit.log"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
evidence="$artifact/official-after-precision-v4-gpu11-v2"
container_artifact="${artifact/\/home\/syk/\/root}"
[[ ! -f "$evidence/postcheck-supervisor.pid" ]] || exit 1
echo "$BASHPID" > "$evidence/postcheck-supervisor.pid"
while [[ ! -f "$evidence/host-launch.exit" ]]; do sleep 5; done
set +e
SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir "$container_artifact/official-after-precision-v4-gpu11-v2" -- bash -c 'artifact="$1"; source /opt/gfx-deps/env.sh; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 64-79 env ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$artifact/canonical-v4-icpx/site" /opt/gfx-deps/venv/bin/python "$artifact/check_canonical_package.py" --module mxfp4-w4a4 --version canonical-v4-icpx --output "$artifact/official-after-precision-v4-gpu11-v2/normal-package-postcheck.json"' bash "$container_artifact" > "$evidence/normal-package-postcheck.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/normal-package-postcheck.exit"
if (( status != 0 )); then tail -c 3000 "$evidence/normal-package-postcheck.log"; exit "$status"; fi
set +e
taskset -c 64-79 python3 "$evidence/audit_results.py" > "$evidence/CPU-result-audit.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/CPU-result-audit.exit"
tail -c 4000 "$evidence/CPU-result-audit.log"
exit "$status"

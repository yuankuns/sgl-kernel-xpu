#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if (($#)); then
  echo 'Fixed B8 performance protocol only; no serving/timing overrides'
  exit 2
fi
printf '%s\n' "$$" > "$root/host-supervisor.pid"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
workdir=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/normal-v4-model-config-perf-v1
set +e
taskset -c 32-47 python3 "$root/run_perf.py" --mode prepare > "$root/prepare.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/prepare.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor.exit"
  exit "$status"
fi
"$wrapper" --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python3 run_perf.py --mode cpu-preflight' > "$root/cpu-preflight.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/cpu-preflight.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor.exit"
  exit "$status"
fi
taskset -c 32-47 python3 "$root/run_perf.py" --mode publish-grant > "$root/grant-publication.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/grant-publication.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor.exit"
  exit "$status"
fi
"$wrapper" --gpu-lock --gpu 6 --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python3 run_perf.py --mode suite' > "$root/gpu-supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/host-supervisor.exit"
tail -25 "$root/gpu-supervisor.log"
exit "$status"

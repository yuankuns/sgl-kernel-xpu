#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if (($#)); then
  echo 'Only fixed B8 model-config accuracy; no serving/timing CLI overrides'
  exit 2
fi
printf '%s\n' "$$" > "$root/host-supervisor-r2.pid"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
workdir=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/normal-v4-model-config-accuracy-v2
set +e
taskset -c 32-47 python3 "$root/run_accuracy_r2.py" --mode prepare > "$root/prepare-r2.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/prepare-r2.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor-r2.exit"
  exit "$status"
fi
"$wrapper" --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python3 run_accuracy_r2.py --mode cpu-prepare' > "$root/cpu-prepare.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/cpu-prepare.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor-r2.exit"
  exit "$status"
fi
taskset -c 32-47 python3 "$root/run_accuracy_r2.py" --mode publish-grant > "$root/grant-publication.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/grant-publication.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/host-supervisor-r2.exit"
  exit "$status"
fi
"$wrapper" --gpu-lock --gpu 6 --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python3 run_accuracy_r2.py --mode suite' > "$root/gpu-supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/host-supervisor-r2.exit"
tail -20 "$root/gpu-supervisor.log"
exit "$status"

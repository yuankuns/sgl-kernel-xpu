#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if (($#)); then
  echo 'accuracy-only: no serving or timing CLI overrides accepted' >&2
  exit 2
fi
taskset -c 32-47 python3 "$root/source_accuracy_v1.py" --mode prepare
printf 'host_shell_pid=%s\n' "$$" > "$root/source-accuracy-v1/host-supervisor.pid"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
workdir=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope
set +e
"$wrapper" --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/source_accuracy_v1.py --mode cpu-preflight' > "$root/source-accuracy-v1/cpu-preflight.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/source-accuracy-v1/cpu-preflight.exit"
if ((status)); then
  printf '%s\n' "$status" > "$root/source-accuracy-v1/host-supervisor.exit"
  exit "$status"
fi
set -e
taskset -c 32-47 python3 "$root/source_accuracy_v1.py" --mode publish-grant > "$root/source-accuracy-v1/grant-publication.log" 2>&1
set +e
"$wrapper" --gpu-lock --gpu 6 --workdir "$workdir" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/source_accuracy_v1.py --mode suite' > "$root/source-accuracy-v1/gpu-supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/source-accuracy-v1/host-supervisor.exit"
tail -20 "$root/source-accuracy-v1/gpu-supervisor.log"
exit "$status"

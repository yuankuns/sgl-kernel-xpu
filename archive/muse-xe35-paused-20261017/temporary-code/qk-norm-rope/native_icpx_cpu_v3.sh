#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
taskset -c 32-47 python3 "$root/native_icpx_cpu_v3.py" --mode prepare
printf 'host_shell_pid=%s\n' "$$" > "$root/native-icpx-cpu-v3/host-supervisor.pid"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope \
  -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 2 \
  bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/native_icpx_cpu_v3.py --mode container' \
  > "$root/native-icpx-cpu-v3/container-supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/native-icpx-cpu-v3/host-supervisor.exit"
tail -35 "$root/native-icpx-cpu-v3/container-supervisor.log"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/framework-patch-gpu.exit' EXIT
exec 9>/tmp/muse-gpu-bench.lock
flock 9
xpu-smi stats -d 0 > artifacts/position/framework-patch-gpu-stats.log 2>&1
xpu-smi ps -d 0 > artifacts/position/framework-patch-gpu-ps.log 2>&1
rg -q 'No data' artifacts/position/framework-patch-gpu-ps.log
rg 'GPU Utilization' artifacts/position/framework-patch-gpu-stats.log
printf 'host=%s date=%s kernel_main=%s sglang_main=%s patched_source=%s aot_lib=%s\n' \
  "$(hostname)" "$(date -Is)" "$(git rev-parse origin/main)" \
  "$(git -C /home/syk/worktrees/sglang/muse-position-triton-split-main rev-parse origin/main)" \
  "$(sha256sum /home/syk/worktrees/sglang/muse-position-triton-split-main/python/sglang/kernels/ops/attention/position.py)" \
  "$(sha256sum build/src/libsgl-ops-sycl-ComputePosition.so)"
for order in old-first patched-first; do
  SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 \
    /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
    --gpu-lock --gpu 0 \
    --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 \
    -- bash -c '
      set -e
      export PYTHONPATH=/workspace/artifacts/position:/workspace/artifacts/position/extension_aot_after
      export POSITION_ORDER="$1"
      cd /workspace
      python artifacts/position/bench_framework_patch.py
    ' bash "$order"
done

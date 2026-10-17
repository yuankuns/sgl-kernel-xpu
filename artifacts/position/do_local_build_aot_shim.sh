#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/local-build-aot-shim.exit' EXIT
[[ $(cat artifacts/position/local-target-after-v2.exit) == 0 ]]
exec 9>/tmp/muse-op-build.lock
flock 9
SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 -- bash -c 'export CXX=icpx MAX_JOBS=1; python /workspace/artifacts/position/build_aot_shim.py'

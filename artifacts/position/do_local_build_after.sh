#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/local-build-after.exit' EXIT
exec 9>/tmp/muse-op-build.lock
flock 9
printf 'host=%s date=%s source=%s\n' "$(hostname)" "$(date -Is)" "$(sha256sum src/sycl/ComputePosition.cpp)"
SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 -- bash -c '
set -e
export CXX=icpx MAX_JOBS=1
export POSITION_SOURCE=/workspace/src/sycl/ComputePosition.cpp
export POSITION_EXTENSION_NAME=position_after
export POSITION_BUILD_DIRECTORY=/workspace/artifacts/position/extension_after
python /workspace/artifacts/position/build_extension.py
'

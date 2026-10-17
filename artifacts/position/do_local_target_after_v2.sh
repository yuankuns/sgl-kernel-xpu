#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/local-target-after-v2.exit' EXIT
exec 9>/tmp/muse-op-build.lock
flock 9
SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 -- bash -c '
set -e
cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=cri -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python -DSKBUILD_SABI_VERSION=3.9 -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-study-20261015/cutlass-e878
cmake --build /workspace/build --target sgl-ops-sycl-ComputePosition -j1
'

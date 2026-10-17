#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
version=$(python3 -S "$art/launch-v3-icpx/settings.py" --version-only)
if [[ -f "$art/$version/source-binary-manifest.json" && ! -r "$art/$version/source-binary-manifest.json" ]]; then
  SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
    /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
    --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35 -- \
    chmod 0644 "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$version/source-binary-manifest.json"
fi
python3 -S "$art/launch-v3-icpx/check_official.py" --preflight-only
exec bash "$art/supervise_xe35.sh" "official-import-$version" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35 -- \
  bash -c 'export FMHA_CANONICAL_VERSION="$1"; site="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$1/site"; export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH="$site"; export LD_LIBRARY_PATH="$site/sgl_kernel:/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; export ZE_AFFINITY_MASK=5; exec taskset -c 16-31 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/launch-v3-icpx/check_official.py' bash "$version"

#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
label=normal-v4-gpu5-accuracy-only-v1
grant=$art/canonical-v4-icpx/gpu5-accuracy-only-barrier-v1.json
test ! -e "$art/$label.handle"
export FMHA_CANONICAL_VERSION=canonical-v4-icpx
export PYTHONDONTWRITEBYTECODE=1
python3 -S "$art/launch-v3-icpx/after_accuracy_only.py" --grant "$grant" --output "$art/$label.json" --guard-only
exec bash "$art/supervise_xe35.sh" "$label" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 5 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha -- \
  bash -c 'export FMHA_CANONICAL_VERSION=canonical-v4-icpx; export PYTHONDONTWRITEBYTECODE=1; site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v4-icpx/site; export PYTHONPATH="$site"; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:$site/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 16-31 /opt/gfx-deps/venv/bin/python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/launch-v3-icpx/after_accuracy_only.py --grant /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v4-icpx/gpu5-accuracy-only-barrier-v1.json --output /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/normal-v4-gpu5-accuracy-only-v1.json'

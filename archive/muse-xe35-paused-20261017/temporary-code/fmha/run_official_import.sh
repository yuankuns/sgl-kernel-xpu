#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
python3 -S "$art/check_official.py" --preflight-only
exec bash "$art/supervise_xe35.sh" official-import env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35 -- \
  bash -c 'site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v2/site; export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH="$site"; export LD_LIBRARY_PATH="$site/sgl_kernel:/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; export ZE_AFFINITY_MASK=5; exec taskset -c 16-31 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/check_official.py'

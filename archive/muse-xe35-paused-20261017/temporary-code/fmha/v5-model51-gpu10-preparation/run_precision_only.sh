#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
runner=$art/v5-model51-gpu10-preparation/precision_v5.py
grant=${1:?fresh main GPU10 precision-only grant path required}
test "$#" = 1
test ! -e "$art/model51-normal-v5-gpu10-accuracy-v1.handle"
export FMHA_CANONICAL_VERSION=canonical-v5-icpx
export PYTHONDONTWRITEBYTECODE=1
container_grant=${grant/#\/home\/syk\//\/root\/}
env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha -- \
  bash -c 'export FMHA_CANONICAL_VERSION=canonical-v5-icpx; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 16-31 python -S /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/v5-model51-gpu10-preparation/precision_v5.py --guard-only --grant "$1"' bash "$container_grant"
exec bash "$art/supervise_xe35.sh" model51-normal-v5-gpu10-accuracy-v1 env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 10 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha -- \
  bash -c 'export FMHA_CANONICAL_VERSION=canonical-v5-icpx; export PYTHONDONTWRITEBYTECODE=1; site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v5-icpx/site; export PYTHONPATH="$site:/root/sglang/python:/root/sglang/sglang_995c774/sglang_meta_common"; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:$site/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; exec taskset -c 16-31 /opt/gfx-deps/venv/bin/python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/v5-model51-gpu10-preparation/precision_v5.py --grant "$1" --output /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/v5-model51-gpu10-preparation/normal-v5-gpu10-accuracy.json' bash "$container_grant"

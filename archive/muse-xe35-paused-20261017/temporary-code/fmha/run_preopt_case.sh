#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
side=${1:?preopt or port}
label=${2:?unique label}
shift 2
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
python3 "$art/bench_preopt_comparison.py" --guard-only "$@"
python3 -S "$art/check_official.py" --preflight-only
python3 -S "$art/check_preopt.py"
case "$side" in
  preopt) site=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017/site-before ;;
  port) site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v2/site ;;
  *) exit 2 ;;
esac
exec bash "$art/supervise_xe35.sh" "$label" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 5 --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35 -- \
  bash -c 'export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH="$1"; export LD_LIBRARY_PATH="$1/sgl_kernel:/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; export FMHA_COMPARISON_PROVIDER="$2"; shift 2; exec taskset -c 16-31 python -u /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/bench_preopt_comparison.py "$@"' bash "$site" "$side" \
  --label "$label" --output "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$label.json" "$@"

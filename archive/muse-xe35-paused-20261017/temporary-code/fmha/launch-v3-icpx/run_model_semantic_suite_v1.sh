#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
export PYTHONDONTWRITEBYTECODE=1
python3 -S "$art/launch-v3-icpx/check_model_semantic_grant.py"
for provider in source port; do
  label=model51-accuracy-$provider-v1
  test ! -e "$art/$label.handle"
  if [[ "$provider" == source ]]; then
    site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/source/site
  else
    test "$(cat "$art/model51-accuracy-source-v1.exit")" = 0
    site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/canonical-v4-icpx/site
  fi
  bash "$art/supervise_xe35.sh" "$label" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
    /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
    --gpu-lock --gpu 5 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha -- \
    bash -c 'export PYTHONDONTWRITEBYTECODE=1; export FMHA_BASELINE_KIND=matched-native-icpx; export PYTHONPATH="$1:/root/sglang/python:/root/sglang/sglang_995c774/sglang_meta_common"; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:$1/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; provider="$2"; exec taskset -c 16-31 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/launch-v3-icpx/model_semantic_accuracy.py --provider "$provider" --output "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/model-config51-v1/$provider-accuracy.json" --grant /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/model-config51-v1/gpu5-accuracy-only-barrier-v1.json' \
    bash "$site" "$provider"
done

#!/usr/bin/env bash
set -euo pipefail
side=${1:?source or port}
name=${2:?run label}
shift 2
root=/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
version=$(python3 -S "$art/launch-v3-icpx/settings.py" --version-only)
kind=${FMHA_BASELINE_KIND:-archived}
[[ "$kind" == archived || "$kind" == matched-native-icpx ]] || exit 2
python3 "$root/benchmark/bench_muse_fmha.py" --guard-only "$@"
python3 -S "$art/launch-v3-icpx/check_official.py" --preflight-only
epoch=${MUSE_SYNC_START_EPOCH:-0}
if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then echo 'MUSE_SYNC_START_EPOCH must be integer seconds' >&2; exit 2; fi
case "$side" in
  source) site=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017/site-after ;;
  port) site="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$version/site" ;;
  *) exit 2 ;;
esac
if [[ "$side" == source ]]; then
  if [[ "$kind" == matched-native-icpx ]]; then
    site=$(python3 -S "$art/launch-v3-icpx/matched_baseline_provenance.py" --role source --site-container-only)
  fi
  python3 -S "$art/launch-v3-icpx/baseline_compiler_provenance.py" --provider source \
    --output "$art/$name.compiler-provenance.json"
fi
python3 -u "$art/launch-v3-icpx/wait_gpu_barrier.py" --phase official
exec bash "$art/supervise_xe35.sh" "$name" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
  /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 5 --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35 -- \
  bash -c 'export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH="$1"; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:$1/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; epoch=$2; export FMHA_CANONICAL_VERSION="$3"; export FMHA_BASELINE_KIND="$4"; shift 4; echo "GPU5 locked; waiting for coordinated epoch=$epoch"; while (( $(date +%s) < epoch )); do sleep 1; done; exec taskset -c 16-31 python -u benchmark/bench_muse_fmha.py "$@"' bash "$site" "$epoch" "$version" "$kind" \
  --label "$name" --output "/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$name.json" "$@"

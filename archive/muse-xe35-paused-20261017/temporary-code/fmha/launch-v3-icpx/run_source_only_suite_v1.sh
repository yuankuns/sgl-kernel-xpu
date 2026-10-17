#!/usr/bin/env bash
set -uo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
root=$art/matched-native-icpx-v1
grant=$root/source-only-gpu5-barrier-v1.json
test "$(cat "$art/matched-native-icpx-resource-v4-supervisor.exit")" = 0 || exit 2
test "$(cat "$art/matched-native-icpx-resource-v4-normal-package-cpu.exit")" = 0 || exit 2
test "$(cat "$art/source-only-resource-v4-all-provider-cpu.exit")" = 0 || exit 2
status=0
for provider in historical-optimized fresh-optimized fresh-preopt; do
  label=source-only-v1-$provider
  output=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/$label.json
  case "$provider" in
    historical-optimized) site=/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017/site-after; kind=archived ;;
    fresh-optimized) site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/source/site; kind=matched-native-icpx ;;
    fresh-preopt) site=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/preopt/site; kind=matched-native-icpx ;;
  esac
  test ! -e "$art/$label.handle" || exit 2
  python3 -S "$art/launch-v3-icpx/source_only_accuracy.py" --provider "$provider" --label "$label" --output "$output" \
    --grant "$grant" --paired-dir "$root/source-only-paired-v1" --guard-only || exit 2
  bash "$art/supervise_xe35.sh" "$label" env SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 \
    /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
    --gpu-lock --gpu 5 --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha -- \
    bash -c 'export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH="$1"; export FMHA_BASELINE_KIND="$2"; export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:$1/sgl_kernel:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"; shift 2; exec taskset -c 16-31 /opt/gfx-deps/venv/bin/python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/launch-v3-icpx/source_only_accuracy.py "$@"' \
    bash "$site" "$kind" --provider "$provider" --label "$label" --output "$output" \
    --grant /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/source-only-gpu5-barrier-v1.json \
    --paired-dir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/matched-native-icpx-v1/source-only-paired-v1
  result=$?
  printf '%s exit=%s\n' "$provider" "$result"
  if (( result != 0 )); then status=1; fi
done
exit "$status"

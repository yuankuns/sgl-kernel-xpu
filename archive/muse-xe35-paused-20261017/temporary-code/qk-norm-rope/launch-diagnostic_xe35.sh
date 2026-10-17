#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for name in BATCHES BATCH_SIZE BATCH SERVING_BATCH SERVING_BATCH_SIZE SGLANG_BATCH_SIZE; do
  if [[ "${!name-8}" != 8 ]]; then
    printf 'scope error: %s must be 8; Muse B8 only\n' "$name" >&2
    exit 2
  fi
done
for specification in ISL:4096 OSL:16 TP:1 EP:1; do
  name="${specification%%:*}"
  expected="${specification#*:}"
  if [[ "${!name-$expected}" != "$expected" ]]; then
    printf 'scope error: %s must be %s\n' "$name" "$expected" >&2
    exit 2
  fi
done
if (($#)); then
  echo 'scope error: no CLI serving overrides accepted' >&2
  exit 2
fi
run="${QK_DIAGNOSTIC_RUN:-diagnostic}"
if [[ ! "$run" =~ ^diagnostic(-[a-zA-Z0-9_-]+)?$ ]]; then
  echo 'Invalid diagnostic run directory' >&2
  exit 2
fi
if [[ -e "$root/$run/supervisor.log" || -e "$root/$run/supervisor.exit" ]]; then
  echo 'Diagnostic run already has evidence; choose a new unique run tag' >&2
  exit 2
fi
if [[ "$run" != diagnostic ]]; then
  mkdir "$root/$run"
  cp "$root/diagnostic/provenance.json" "$root/$run/provenance.json"
fi
python3 "$root/wait_gpu_barrier.py" --phase diagnostic
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 6 \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35 -- \
  bash -c 'export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}; export PYTHONDONTWRITEBYTECODE=1; export QK_DIAGNOSTIC_ROOT="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/$1"; exec taskset -c 32-47 python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/run_suite.py' _ "$run" \
  >"$root/$run/supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" >"$root/$run/supervisor.exit"
exit "$status"

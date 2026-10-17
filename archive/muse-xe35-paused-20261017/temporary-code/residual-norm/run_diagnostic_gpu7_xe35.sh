#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
for name in BATCHES BATCH SERVING_BATCH BATCH_SIZE SGLANG_BATCH_SIZE SGLANG_MUSE_BATCH_SIZE MUSE_BATCH_SIZE SERVING_BATCH_SIZE MUSE_BS BS; do
  if [[ "${!name-8}" != 8 ]]; then
    printf 'B8-only: %s must be 8\n' "$name" >&2
    exit 2
  fi
done
for specification in ISL:4096 OSL:16 TP:1 EP:1; do
  name="${specification%%:*}"
  expected="${specification#*:}"
  if [[ "${!name-$expected}" != "$expected" ]]; then
    printf 'production scope: %s must be %s\n' "$name" "$expected" >&2
    exit 2
  fi
done
if (( $# != 0 )); then
  printf '%s\n' 'B8-only diagnostic launcher accepts no scope overrides' >&2
  exit 2
fi
while [[ ! -f "$artifact/diagnostic_native_build.exit" ]]; do
  sleep 2
done
test "$(cat "$artifact/diagnostic_native_build.exit")" = 0
python3 "$artifact/wait_gpu_barrier.py" --phase diagnostic
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  --gpu-lock --gpu 7 taskset -c 48-63 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/validate_diagnostic_container_xe35.sh \
  >"$artifact/diagnostic_gpu7.log" 2>&1
status=$?
echo "$status" >"$artifact/diagnostic_gpu7.exit"
tail -n 12 "$artifact/diagnostic_gpu7.log"
exit "$status"

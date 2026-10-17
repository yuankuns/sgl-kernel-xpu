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
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
export PYTHONDONTWRITEBYTECODE=1
python3 - "$root/official-release-policy.json" <<'PY'
import json
import sys

policy = json.load(open(sys.argv[1]))
if not policy['official_gpu_launch_authorized']:
    print('Official GPU6 launch paused: require genuine complete v4 icpx package and new main grant', file=sys.stderr)
    raise SystemExit(2)
PY
version="${QK_CANONICAL_VERSION:-canonical-v4-icpx}"
if [[ "$version" != canonical-v4-icpx ]]; then
  echo 'Old versions diagnostic only: canonical-v4-icpx and new main GPU6 grant required' >&2
  exit 2
fi
run="${QK_OFFICIAL_RUN:-official-v4-icpx}"
if [[ ! "$run" =~ ^official-[a-zA-Z0-9._-]+$ ]]; then
  echo 'invalid official run tag' >&2
  exit 2
fi
runtime="$root/$version/$run"
if [[ -e "$runtime" || -e "$root/$version-$run-preflight.log" ]]; then
  echo 'Official run already has evidence; choose a new unique run tag' >&2
  exit 2
fi
set +e
python3 "$root/canonical_check.py" --canonical-version "$version" --runtime-root "$runtime" > "$root/$version-$run-preflight.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/$version-$run-preflight.exit"
set -e
if ((status)); then
  cat "$root/$version-$run-preflight.log" >&2
  exit "$status"
fi
python3 "$root/audit_preopt_evidence.py" > "$runtime/preopt-provider-preflight.log" 2>&1
export QK_OFFICIAL_ROOT="${runtime/#\/home\/syk/\/root}"
unset QK_DIAGNOSTIC_ROOT
python3 "$root/wait_gpu_barrier.py" --phase official --canonical-version "$version"
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --gpu-lock --gpu 6 \
  --workdir /root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35 -- \
  bash -c 'export QK_OFFICIAL_ROOT="$1"; export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35; export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:/opt/gfx-deps/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}; export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 bash -c '\''python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/run_suite.py && python /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/qk-norm-rope/run_preopt.py'\''' qk-official "$QK_OFFICIAL_ROOT" \
  > "$runtime/supervisor.log" 2>&1
status=$?
printf '%s\n' "$status" > "$runtime/supervisor.exit"
tail -50 "$runtime/supervisor.log"
exit "$status"

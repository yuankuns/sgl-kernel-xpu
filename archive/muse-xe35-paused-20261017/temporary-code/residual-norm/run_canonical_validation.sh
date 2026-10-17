#!/usr/bin/env bash
set -euo pipefail
artifact=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm
for name in BATCHES BATCH SERVING_BATCH BATCH_SIZE SGLANG_BATCH_SIZE SGLANG_MUSE_BATCH_SIZE MUSE_BATCH_SIZE SERVING_BATCH_SIZE MUSE_BS BS; do
  [[ "${!name-8}" == 8 ]] || exit 2
done
for specification in ISL:4096 OSL:16 TP:1 EP:1; do
  name="${specification%%:*}"
  [[ "${!name-${specification#*:}}" == "${specification#*:}" ]] || exit 2
done
[[ "$#" -le 1 ]] || exit 2
version="${1:-canonical-v2}"
[[ "$version" == canonical-v2 ]] || exit 2
python3 -B - "$artifact" "$version" <<'PY'
import json
import pathlib
import sys
import time
root = pathlib.Path(sys.argv[1])
version = sys.argv[2]
freeze = json.loads((root / 'canonical-source-freeze-v2.json').read_text())
grant = json.loads((root / f'gpu7-{version}-validation-barrier.json').read_text())
assert grant['gpu'] == 7 and grant['coordinator_release'] is True
assert grant['phase'] == f'residual-{version}-validation'
assert grant['source_digest'] == freeze['source_digest']
assert (root / version / 'source-binary-manifest.json').is_file()
epoch = float(grant['start_epoch'])
assert epoch > 0
time.sleep(max(0, epoch - time.time()))
PY
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  --gpu-lock --gpu 7 taskset -c 48-63 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/validate_canonical_container.sh "$version" \
  > "$artifact/${version}_validation.log" 2>&1
status=$?
echo "$status" > "$artifact/${version}_validation.exit"
exit "$status"

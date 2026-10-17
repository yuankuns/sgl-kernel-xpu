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
version="${1:-canonical-v4-icpx}"
[[ "$version" == canonical-v4-icpx ]] || exit 2
exec 9>"$artifact/.$version-validation.lock"
flock -n 9
test ! -e "$artifact/canonical-validation-${version#canonical-}"
python3 -B - "$artifact" "$version" <<'PY'
import json
import hashlib
import math
import pathlib
import runpy
import sys
import time
root = pathlib.Path(sys.argv[1])
version = sys.argv[2]
freeze = json.loads((root / f'canonical-source-freeze-{version.removeprefix("canonical-")}.json').read_text())
grant = json.loads((root / f'gpu7-{version}-validation-barrier.json').read_text())
assert grant['gpu'] == 7 and grant['coordinator_release'] is True
assert grant['cpu_workers_drained'] is True
assert grant['wave_required_v4_packages_and_SOURCE_gates_passed'] is True
assert grant['phase'] == f'residual-{version}-validation'
assert grant['source_digest'] == freeze['source_digest']
manifest_path = root / version / 'source-binary-manifest.json'
policy_path = root / version / 'compiler-policy-verified.json'
assert grant['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert grant['compiler_policy_sha256'] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
checker = runpy.run_path(str(root / 'check_canonical_package_v3.py'))
checker['provenance'](version)
checker['before_provenance']()
assert grant['before_manifest_sha256'] == hashlib.sha256((root / 'source-before-icpx-bounded-v1/native-package-manifest.json').read_bytes()).hexdigest()
epoch = float(grant['start_epoch'])
post = json.loads((root / version / 'source-readonly-postverify.json').read_text())
assert math.isfinite(epoch) and epoch >= post['epoch']
time.sleep(max(0, epoch - time.time()))
PY
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
set +e
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  -- bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/build-lane.sh 3 \
  /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/source-full-venv/bin/python -B \
  /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/check_canonical_package_v3.py \
  --build-version "$version" --mode provenance \
  > "$artifact/${version}_CPU_preflight.log" 2>&1
status=$?
echo "$status" > "$artifact/${version}_CPU_preflight.exit"
[[ "$status" == 0 ]] || exit "$status"
/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \
  --workdir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm \
  --gpu-lock --gpu 7 taskset -c 48-63 bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/validate_canonical_container_v3.sh "$version" \
  > "$artifact/${version}_validation.log" 2>&1
status=$?
echo "$status" > "$artifact/${version}_validation.exit"
exit "$status"

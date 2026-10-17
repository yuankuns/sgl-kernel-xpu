#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
phase="$1"
run="$2"
version="${4:-canonical-v4-icpx}"
[[ "$version" == canonical-v4-icpx ]] || exit 2
[[ "$phase" == accuracy || "$phase" == performance ]] || exit 2
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
if [[ "$phase" == performance ]]; then
  [[ "${3:-}" =~ ^[a-f0-9]{64}$ ]] || exit 2
  export NVFP4_GRANTED_MANIFEST_SHA256="$3"
else
  unset NVFP4_GRANTED_MANIFEST_SHA256
fi
canonical="$artifact/$version"
python="$artifact/target-venv/bin/python"
export BATCHES=8 ISL=4096 OSL=16 TP=1 EP=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$canonical/site:$artifact"
run_step() {
  name="$1"
  shift
  [[ ! -e "$artifact/$name.log" ]] || exit 2
  set +e
  "$@" >"$artifact/$name.log" 2>&1
  status=$?
  printf '%s\n' "$status" >"$artifact/$name.exit"
  tail -n 8 "$artifact/$name.log"
  set -e
  return "$status"
}
run_step "$run-package-runtime" "$python" "$artifact/canonical_check.py" --version "$version" --runtime
if [[ "$phase" == accuracy ]]; then
  run_step "$run-original-tests" "$python" -m pytest -q -p diagnostic_cooldown \
    /root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35/tests/test_nvfp4_woq_moe.py \
    --junitxml="$artifact/$run-original-tests.xml"
  "$python" - "$artifact/$run-original-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as etree
suites = list(etree.parse(sys.argv[1]).getroot().iter('testsuite'))
assert sum(int(suite.attrib['tests']) for suite in suites) == 50
assert all(int(suite.attrib.get(key, 0)) == 0 for suite in suites for key in ('skipped', 'failures', 'errors'))
PY
fi
suites=(production)
if [[ "$phase" == accuracy ]]; then suites=(tails production); fi
for suite in "${suites[@]}"; do
  controls=()
  rounds=3
  if [[ "$phase" == accuracy ]]; then controls+=(--accuracy-only); rounds=1; fi
  if [[ "$suite" == tails ]]; then controls+=(--tail-only); fi
  labels=(port source source port)
  for ((round=0; round<rounds; round++)); do
    for position in 0 1 2 3; do
      label="${labels[$position]}"
      export PYTHONPATH="$canonical/site:$artifact"
      if [[ "$label" == source ]]; then export PYTHONPATH="$artifact/native-diagnostic/source/site:$artifact"; fi
      run_step "$run-$suite-r$round-p$position-$label" "$python" "$artifact/run_canonical_case.py" \
        "$label" --version "$version" --run "$run-$suite" --round "$round" --phase "$position" "${controls[@]}"
    done
  done
  run_step "$run-$suite-analysis" "$python" "$artifact/analyze_canonical_results.py" --version "$version" --phase "$phase" --run "$run-$suite"
done

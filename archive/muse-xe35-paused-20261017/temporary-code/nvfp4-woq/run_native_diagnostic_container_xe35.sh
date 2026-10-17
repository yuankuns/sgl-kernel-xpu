#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
export BATCHES=8 ISL=4096 OSL=16 TP=1 EP=1
python="$artifact/target-venv/bin/python"
run="${1:-firstpass}"
mode="${2:-all}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$mode" == all || "$mode" == tails || "$mode" == production || "$mode" == tests || "$mode" == kernels ]] || exit 2
test -f "$artifact/diagnostic.READY"
run_step() {
  name="$1"
  shift
  set +e
  "$@" >"$artifact/$name.log" 2>&1
  status=$?
  printf '%s\n' "$status" >"$artifact/$name.exit"
  tail -n 8 "$artifact/$name.log"
  set -e
  return "$status"
}
if [[ "$mode" == all || "$mode" == tests ]]; then
  export PYTHONPATH="$artifact/native-diagnostic/port/site:$artifact"
  run_step "$run-original-tests-port" "$python" -m pytest -q --run-long-tests -p diagnostic_cooldown \
    /root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014/tests/test_nvfp4_woq_moe.py \
    --junitxml="$artifact/$run-original-tests-port.xml"
fi
for suite in tails production; do
  [[ "$mode" == all || "$mode" == kernels || "$mode" == "$suite" ]] || continue
  controls=()
  rounds=3
  if [[ "$suite" == tails ]]; then controls+=(--tail-only); rounds=1; fi
  labels=(port source source port)
  for ((round=0; round<rounds; round++)); do
    for phase in 0 1 2 3; do
      label="${labels[$phase]}"
      export PYTHONPATH="$artifact/native-diagnostic/$label/site:$artifact"
      run_step "$run-$suite-r$round-p$phase-$label" "$python" "$artifact/run_native_case.py" \
        "$label" --run "$run-$suite" --round "$round" --phase "$phase" "${controls[@]}"
    done
  done
  run_step "$run-$suite-analysis" "$python" "$artifact/analyze_native_diagnostic.py" "$run-$suite"
done

#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq
export LD_LIBRARY_PATH=/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}
export BATCHES=8 ISL=4096 OSL=16 TP=1 EP=1
python="$artifact/target-venv/bin/python"
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
export PYTHONPATH="$artifact/native-diagnostic/port/site:$artifact"
run_step grf512-accuracy-only-original-tests "$python" -m pytest -q --run-long-tests -p diagnostic_cooldown \
  /root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014/tests/test_nvfp4_woq_moe.py \
  --junitxml="$artifact/grf512-accuracy-only-original-tests.xml"
labels=(port source source port)
for phase in 0 1 2 3; do
  label="${labels[$phase]}"
  export PYTHONPATH="$artifact/native-diagnostic/$label/site:$artifact"
  run_step "grf512-accuracy-only-tails-p$phase-$label" "$python" "$artifact/run_native_case.py" \
    "$label" --run grf512-accuracy-only-tails --round 0 --phase "$phase" --tail-only
done
run_step grf512-accuracy-only-tails-analysis "$python" "$artifact/analyze_gpu9_tail_accuracy.py"

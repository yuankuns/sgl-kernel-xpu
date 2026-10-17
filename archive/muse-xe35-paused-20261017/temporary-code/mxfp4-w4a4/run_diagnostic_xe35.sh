#!/usr/bin/env bash
set -euo pipefail
module="${1:?module required}"
phase="${2:-baab}"
case "$module" in mxfp4-w4a4|routing-quant) ;; *) exit 2 ;; esac
case "$phase" in baab|tests|all) ;; *) exit 2 ;; esac
artifact="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/$module"
cd "$artifact"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
printf 'GPU8 using %s actual port native diagnostic, phase=%s; NOT official full-package pass\n' "$module" "$phase"
if [[ "$phase" != baab ]]; then
  if [[ "$module" == mxfp4-w4a4 ]]; then
    source=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014
    paths=("$source/tests/test_mxfp4_moe_w4a4.py")
    options=(--run-long-tests)
  else
    source=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014
    paths=("$source/tests/test_moe_routing.py" "$source/tests/test_per_token_group_quant_mxfp4.py")
    options=()
  fi
  set +e
  "$artifact/diagnostic-venv/bin/python" -m pytest -p no:cacheprovider -p cooldown -v "${options[@]}" "${paths[@]}" > "$artifact/diagnostic-tests.log" 2>&1
  status=$?
  echo "$status" > "$artifact/diagnostic-tests.exit"
  set -e
  if [[ "$status" != 0 ]]; then tail -c 5000 "$artifact/diagnostic-tests.log"; exit "$status"; fi
  if [[ "$phase" == tests ]]; then exit 0; fi
fi
for offset in 0 4; do
  for relative_round in 0 1 2 3; do
    round_id=$((offset + relative_round))
    implementation=port
    environment=diagnostic
    options=(--port-original-abi)
    if [[ "$relative_round" == 0 || "$relative_round" == 3 ]]; then
      implementation=source
      environment=source
      options=()
    fi
    iterations=30
    if [[ "$offset" == 4 ]]; then iterations=100; fi
    set +e
    "$artifact/$environment-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --iterations "$iterations" --evidence-prefix diagnostic- "${options[@]}" > "$artifact/diagnostic-BAAB-$round_id-$implementation.log" 2>&1
    status=$?
    echo "$status" > "$artifact/diagnostic-BAAB-$round_id-$implementation.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 4500 "$artifact/diagnostic-BAAB-$round_id-$implementation.log"; exit "$status"; fi
  done
  set +e
  "$artifact/diagnostic-venv/bin/python" summarize.py --offset "$offset" --prefix diagnostic- > "$artifact/diagnostic-summary-$offset.log" 2>&1
  status=$?
  echo "$status" > "$artifact/diagnostic-summary-$offset.exit"
  set -e
  cat "$artifact/diagnostic-summary-$offset.log"
  if [[ "$status" == 0 ]]; then exit 0; fi
  if [[ "$status" != 5 ]]; then exit "$status"; fi
done
exit 5

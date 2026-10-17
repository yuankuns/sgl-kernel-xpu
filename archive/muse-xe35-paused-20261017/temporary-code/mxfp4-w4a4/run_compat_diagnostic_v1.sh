#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
cd "$artifact"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
printf 'GPU8 W4 source-preserving dependency adaptation; precision ONLY, no performance work; historical strict005 FAIL retained; NOT official package pass\n'
for offset in 0; do
  for relative_round in 0 1 2 3; do
    round_id=$((offset + relative_round))
    implementation=port
    environment=diagnostic
    if [[ "$relative_round" == 0 || "$relative_round" == 3 ]]; then
      implementation=source
      environment=fresh-source
    fi
    set +e
    "$artifact/$environment-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --port-original-abi --evidence-prefix compat-diagnostic- --input-forensics --precision-only > "$artifact/compat-diagnostic-BAAB-$round_id-$implementation.log" 2>&1
    status=$?
    echo "$status" > "$artifact/compat-diagnostic-BAAB-$round_id-$implementation.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 5000 "$artifact/compat-diagnostic-BAAB-$round_id-$implementation.log"; exit "$status"; fi
  done
  set +e
  "$artifact/diagnostic-venv/bin/python" summarize.py --offset "$offset" --prefix compat-diagnostic- --precision-only > "$artifact/compat-diagnostic-summary-$offset.log" 2>&1
  status=$?
  echo "$status" > "$artifact/compat-diagnostic-summary-$offset.exit"
  set -e
  cat "$artifact/compat-diagnostic-summary-$offset.log"
  if [[ "$status" == 0 ]]; then
    set +e
    "$artifact/diagnostic-venv/bin/python" -m pytest -p no:cacheprovider -p cooldown -v --run-long-tests /root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/tests/test_mxfp4_moe_w4a4.py > "$artifact/compat-diagnostic-original-tests.log" 2>&1
    status=$?
    echo "$status" > "$artifact/compat-diagnostic-original-tests.exit"
    tail -c 5000 "$artifact/compat-diagnostic-original-tests.log"
    exit "$status"
  fi
  if [[ "$status" != 5 ]]; then exit "$status"; fi
done
exit 5

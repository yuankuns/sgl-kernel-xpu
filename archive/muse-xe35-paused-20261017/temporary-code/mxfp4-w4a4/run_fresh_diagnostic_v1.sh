#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
cd "$artifact"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
printf 'GPU8 fresh original source matched compiler diagnostic; old source-to-port FAIL retained, not official package pass\n'
for offset in 0 4; do
  for relative_round in 0 1 2 3; do
    round_id=$((offset + relative_round))
    implementation=port
    environment=diagnostic
    if [[ "$relative_round" == 0 || "$relative_round" == 3 ]]; then
      implementation=source
      environment=fresh-source
    fi
    iterations=30
    if [[ "$offset" == 4 ]]; then iterations=100; fi
    set +e
    "$artifact/$environment-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --iterations "$iterations" --port-original-abi --evidence-prefix fresh-diagnostic- --input-forensics > "$artifact/fresh-diagnostic-BAAB-$round_id-$implementation.log" 2>&1
    status=$?
    echo "$status" > "$artifact/fresh-diagnostic-BAAB-$round_id-$implementation.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 5000 "$artifact/fresh-diagnostic-BAAB-$round_id-$implementation.log"; exit "$status"; fi
  done
  set +e
  "$artifact/diagnostic-venv/bin/python" summarize.py --offset "$offset" --prefix fresh-diagnostic- > "$artifact/fresh-diagnostic-summary-$offset.log" 2>&1
  status=$?
  echo "$status" > "$artifact/fresh-diagnostic-summary-$offset.exit"
  set -e
  cat "$artifact/fresh-diagnostic-summary-$offset.log"
  if [[ "$status" == 0 ]]; then exit 0; fi
  if [[ "$status" != 5 ]]; then exit "$status"; fi
done
exit 5

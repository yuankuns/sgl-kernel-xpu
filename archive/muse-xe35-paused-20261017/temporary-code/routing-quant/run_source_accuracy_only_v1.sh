#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
helpers=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
evidence="$artifact/historical-fresh-source-precision-v1"
historical="$artifact/historical-source-normal-v1/site"
fresh="$artifact/fresh-source-icpx-native-v2/site"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
python=/opt/gfx-deps/venv/bin/python
for round_id in 0 1 2 3; do
  kind=fresh
  site="$fresh"
  if [[ "$round_id" == 0 || "$round_id" == 3 ]]; then kind=historical; site="$historical"; fi
  set +e
  PYTHONPATH="$site:$helpers" "$python" "$artifact/source_accuracy_only.py" --kind "$kind" --round "$round_id" --evidence "$evidence" > "$evidence/round-$round_id-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/round-$round_id-$kind.exit"
  if (( status != 0 )); then tail -c 5000 "$evidence/round-$round_id-$kind.log"; exit "$status"; fi
done
set +e
"$python" "$artifact/compare_source_accuracy.py" --evidence "$evidence" > "$evidence/source-paired-precision.log" 2>&1
status=$?
set -e
echo "$status" > "$evidence/source-paired-precision.exit"
if (( status != 0 )); then tail -c 5000 "$evidence/source-paired-precision.log"; exit "$status"; fi
source=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014
for kind in historical fresh; do
  site="$fresh"
  if [[ "$kind" == historical ]]; then site="$historical"; fi
  set +e
  PYTHONPATH="$site:$helpers" "$python" -m pytest -p no:cacheprovider -p cooldown -v "$source/tests/test_moe_routing.py" "$source/tests/test_per_token_group_quant_mxfp4.py" "$source/tests/test_topk_sigmoid.py" "$source/tests/test_routing_benchmark_scope.py" > "$evidence/original-tests-$kind.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$evidence/original-tests-$kind.exit"
  tail -c 1800 "$evidence/original-tests-$kind.log"
  if (( status != 0 )); then exit "$status"; fi
done

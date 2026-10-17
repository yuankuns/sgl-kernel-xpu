#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
for implementation in source port; do
  python="$artifact/$implementation-venv/bin/python"
  if [[ "$implementation" == source ]]; then
    source_tree=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014
    tests=(test_moe_routing.py test_per_token_group_quant_mxfp4.py test_topk_sigmoid.py test_routing_benchmark_scope.py)
  else
    source_tree=/root/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35
    tests=(test_moe_routing.py test_per_token_group_quant_mxfp4_routing.py test_topk_sigmoid_routing.py test_routing_benchmark_scope.py test_topk_sigmoid.py test_per_token_group_quant_mxfp4.py)
  fi
  paths=()
  for test in "${tests[@]}"; do paths+=("$source_tree/tests/$test"); done
  set +e
  "$python" -m pytest -p no:cacheprovider -p cooldown -v "${paths[@]}" > "$artifact/$implementation-tests.log" 2>&1
  status=$?
  echo "$status" > "$artifact/$implementation-tests.exit"
  set -e
  if [[ "$status" != 0 ]]; then
    tail -c 5000 "$artifact/$implementation-tests.log"
    exit "$status"
  fi
done
"$artifact/port-venv/bin/python" -c 'import hashlib,json,pathlib,sgl_kernel; root=pathlib.Path(sgl_kernel.__file__).parent; print(json.dumps({"package":str(root),"binaries":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.so"))}},indent=2))' > "$artifact/port-binary-provenance.json"
for offset in 0 4; do
  for relative_round in 0 1 2 3; do
    round_id=$((offset + relative_round))
    implementation=port
    if [[ "$relative_round" == 0 || "$relative_round" == 3 ]]; then implementation=source; fi
    iterations=30
    if [[ "$offset" == 4 ]]; then iterations=100; fi
    set +e
    "$artifact/$implementation-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --iterations "$iterations" > "$artifact/BAAB-$round_id-$implementation.log" 2>&1
    status=$?
    echo "$status" > "$artifact/BAAB-$round_id-$implementation.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 4500 "$artifact/BAAB-$round_id-$implementation.log"; exit "$status"; fi
  done
  set +e
  "$artifact/port-venv/bin/python" summarize.py --offset "$offset" > "$artifact/summary-$offset.log" 2>&1
  status=$?
  echo "$status" > "$artifact/summary-$offset.exit"
  set -e
  cat "$artifact/summary-$offset.log"
  if [[ "$status" == 0 ]]; then exit 0; fi
  if [[ "$status" != 5 ]]; then exit "$status"; fi
done
exit 5

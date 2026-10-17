#!/usr/bin/env bash
set -euo pipefail
module="${1:?module required}"
phase="${2:?phase required}"
version="${3:?canonical version required}"
[[ "$version" =~ ^canonical-v[1-9][0-9]*-icpx$ ]] || exit 125
[[ "$version" != canonical-v3-icpx ]] || exit 125
case "$module" in mxfp4-w4a4|routing-quant) ;; *) exit 2 ;; esac
case "$phase" in precision|performance) ;; *) exit 2 ;; esac
artifact="/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/$module"
site="$artifact/$version/site"
prefix="$version-$phase-"
cd "$artifact"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
printf 'AFTER normal canonical package %s %s %s; real original source baseline; no diagnostic forwards/native replacement\n' "$module" "$version" "$phase"
if [[ "$phase" == precision ]]; then
  for implementation in source port; do
    options=()
    if [[ "$module" == mxfp4-w4a4 ]]; then
      options=(--run-long-tests)
      tests=(test_mxfp4_moe_w4a4.py)
      source_tree=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014
    else
      tests=(test_moe_routing.py test_per_token_group_quant_mxfp4.py test_topk_sigmoid.py test_routing_benchmark_scope.py)
      source_tree=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014
    fi
    if [[ "$implementation" == port ]]; then
      source_tree="/root/worktrees/sgl-kernel-xpu-up/pr/muse-$module-xe35"
      if [[ "$module" == routing-quant ]]; then tests=(test_moe_routing.py test_per_token_group_quant_mxfp4_routing.py test_topk_sigmoid_routing.py test_routing_benchmark_scope.py test_topk_sigmoid.py test_per_token_group_quant_mxfp4.py); fi
      environment=(env "PYTHONPATH=$site")
    else
      environment=(env -u PYTHONPATH)
    fi
    paths=()
    for test in "${tests[@]}"; do paths+=("$source_tree/tests/$test"); done
    set +e
    "${environment[@]}" "$artifact/$implementation-venv/bin/python" -m pytest -p no:cacheprovider -p cooldown -v "${options[@]}" "${paths[@]}" > "$artifact/$prefix$implementation-tests.log" 2>&1
    status=$?
    echo "$status" > "$artifact/$prefix$implementation-tests.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 5000 "$artifact/$prefix$implementation-tests.log"; exit "$status"; fi
  done
fi
offsets=(0 4)
extra=()
if [[ "$phase" == precision ]]; then offsets=(0); extra=(--precision-only); fi
if [[ "$module" == mxfp4-w4a4 ]]; then extra+=(--input-forensics); fi
for offset in "${offsets[@]}"; do
  for relative_round in 0 1 2 3; do
    round_id=$((offset + relative_round))
    implementation=port
    environment=(env "PYTHONPATH=$site")
    if [[ "$relative_round" == 0 || "$relative_round" == 3 ]]; then implementation=source; environment=(env -u PYTHONPATH); fi
    iterations=30
    if [[ "$offset" == 4 ]]; then iterations=100; fi
    set +e
    "${environment[@]}" "$artifact/$implementation-venv/bin/python" compare_production.py --implementation "$implementation" --round "$round_id" --iterations "$iterations" --evidence-prefix "$prefix" "${extra[@]}" > "$artifact/${prefix}BAAB-$round_id-$implementation.log" 2>&1
    status=$?
    echo "$status" > "$artifact/${prefix}BAAB-$round_id-$implementation.exit"
    set -e
    if [[ "$status" != 0 ]]; then tail -c 5000 "$artifact/${prefix}BAAB-$round_id-$implementation.log"; exit "$status"; fi
  done
  summary_options=()
  if [[ "$phase" == precision ]]; then summary_options=(--precision-only); fi
  set +e
  env "PYTHONPATH=$site" "$artifact/port-venv/bin/python" summarize.py --offset "$offset" --prefix "$prefix" "${summary_options[@]}" > "$artifact/${prefix}summary-$offset.log" 2>&1
  status=$?
  echo "$status" > "$artifact/${prefix}summary-$offset.exit"
  set -e
  cat "$artifact/${prefix}summary-$offset.log"
  if [[ "$status" == 0 ]]; then exit 0; fi
  if [[ "$status" != 5 || "$phase" == precision ]]; then exit "$status"; fi
done
exit 5

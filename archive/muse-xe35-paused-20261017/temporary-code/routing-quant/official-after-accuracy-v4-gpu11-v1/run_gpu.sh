#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/official-after-accuracy-v4-gpu11-v1"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$directory:$artifact/canonical-v4-icpx/site:$artifact:$artifact/../mxfp4-w4a4"
echo "$BASHPID" > "$directory/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$directory/container-affinity.txt"
/opt/gfx-deps/venv/bin/python -c 'import os,torch,sgl_kernel; assert sorted(os.sched_getaffinity(0))==list(range(72,80)); assert os.environ["ZE_AFFINITY_MASK"]=="11"; assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(); print(sgl_kernel.__file__,flush=True)'
overall=0
set +e
/opt/gfx-deps/venv/bin/python "$directory/production_accuracy.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1 --kind after --round 0 --evidence "$directory" > "$directory/six-production-shapes.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/six-production-shapes.exit"
if (( status != 0 )); then overall="$status"; fi
if (( status == 0 )); then
  set +e
  /opt/gfx-deps/venv/bin/python "$directory/compare_full_outputs.py" > "$directory/full-SOURCE-AFTER-comparison.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$directory/full-SOURCE-AFTER-comparison.exit"
  if (( status != 0 )); then overall="$status"; fi
fi
set +e
/opt/gfx-deps/venv/bin/python "$directory/fixture_accuracy.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1 > "$directory/independent-four-component-accuracy.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/independent-four-component-accuracy.exit"
if (( status != 0 )); then overall="$status"; fi
set +e
/opt/gfx-deps/venv/bin/python -m pytest -p no:cacheprovider -p cooldown -p case_record -v --showlocals "$directory/tests/test_moe_routing.py" "$directory/tests/test_per_token_group_quant_mxfp4.py" "$directory/tests/test_topk_sigmoid.py" "$directory/tests/test_routing_benchmark_scope.py" > "$directory/original-adapted-full-1300.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/original-adapted-full-1300.exit"
if (( status != 0 )); then overall="$status"; fi
tail -c 4500 "$directory/original-adapted-full-1300.log"
exit "$overall"

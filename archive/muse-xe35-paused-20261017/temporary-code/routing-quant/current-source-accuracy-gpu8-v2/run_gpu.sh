#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/current-source-accuracy-gpu8-v2"
source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$directory:$artifact/fresh-current-source-icpx-native-v3/site:$artifact:$artifact/../mxfp4-w4a4"
echo "$BASHPID" > "$directory/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$directory/container-affinity.txt"
set +e
/opt/gfx-deps/venv/bin/python "$directory/verify_runtime_cpu.py" > "$directory/preGPU-normal-source-verification.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/preGPU-normal-source-verification.exit"
((status==0)) || exit "$status"
/opt/gfx-deps/venv/bin/python -c 'import os,torch,sgl_kernel,time; assert sorted(os.sched_getaffinity(0))==list(range(64,80)); assert os.environ["ZE_AFFINITY_MASK"]=="8"; assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(); torch.xpu.synchronize(); time.sleep(5)'
run_phase() {
  local phase="$1"
  shift
  set +e
  "$@" > "$directory/$phase.log" 2>&1
  status=$?
  set -e
  echo "$status" > "$directory/$phase.exit"
  ((status==0)) || exit "$status"
}
export CURRENT_SOURCE_TEST_PHASE=original-silu-and-mul-opmath-14
run_phase original-silu-and-mul-opmath-14 /opt/gfx-deps/venv/bin/python -m pytest -p no:cacheprovider -p cooldown -p case_record_v2 -x -v --showlocals "$source_tests/test_silu_and_mul_opmath.py"
export CURRENT_SOURCE_TEST_PHASE=original-full-1300
run_phase original-full-1300 /opt/gfx-deps/venv/bin/python -m pytest -p no:cacheprovider -p cooldown -p case_record_v2 -x -v --showlocals "$source_tests/test_moe_routing.py" "$source_tests/test_per_token_group_quant_mxfp4.py" "$source_tests/test_topk_sigmoid.py" "$source_tests/test_routing_benchmark_scope.py"
run_phase six-production-shapes /opt/gfx-deps/venv/bin/python "$directory/six_production_shapes.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1 --kind fresh --round 0 --evidence "$directory"
run_phase independent-four-component-accuracy /opt/gfx-deps/venv/bin/python "$directory/fixture_accuracy.py" --serving-batch 8 --isl 4096 --osl 16 --tp 1 --ep 1

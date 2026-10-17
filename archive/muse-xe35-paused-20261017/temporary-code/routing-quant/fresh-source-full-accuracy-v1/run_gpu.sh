#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
directory="$artifact/fresh-source-full-accuracy-v1"
source_tests=/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$directory:$artifact/fresh-source-icpx-native-v2/site:$artifact"
echo "$BASHPID" > "$directory/container-runtime.pid"
grep Cpus_allowed_list /proc/self/status > "$directory/container-affinity.txt"
/opt/gfx-deps/venv/bin/python -c 'import os,torch,sgl_kernel; assert sorted(os.sched_getaffinity(0))==list(range(64,72)); assert os.environ["ZE_AFFINITY_MASK"]=="8"; assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(); print(sgl_kernel.__file__,flush=True)'
set +e
/opt/gfx-deps/venv/bin/python -m pytest -p no:cacheprovider -p cooldown -p case_record -v --showlocals "$source_tests/test_moe_routing.py" "$source_tests/test_per_token_group_quant_mxfp4.py" "$source_tests/test_topk_sigmoid.py" "$source_tests/test_routing_benchmark_scope.py" > "$directory/original-full-1300.log" 2>&1
status=$?
set -e
echo "$status" > "$directory/original-full-1300.exit"
tail -c 4500 "$directory/original-full-1300.log"
exit "$status"

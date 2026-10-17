#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/local-int32-aot-after.exit' EXIT
for required in local-build-aot-shim local-int32-after; do
  while [[ ! -f "artifacts/position/$required.exit" ]]; do sleep 5; done
  [[ $(cat "artifacts/position/$required.exit") == 0 ]]
done
exec 9>/tmp/muse-gpu-bench.lock
flock 9
xpu-smi stats -d 0 > artifacts/position/local-int32-aot-after-gpu-stats.log 2>&1
xpu-smi ps -d 0 > artifacts/position/local-int32-aot-after-gpu-ps.log 2>&1
rg -q 'No data' artifacts/position/local-int32-aot-after-gpu-ps.log
printf 'host=%s date=%s production_lib=%s\n' "$(hostname)" "$(date -Is)" "$(sha256sum build/src/libsgl-ops-sycl-ComputePosition.so)"
SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 0 --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 -- bash -c '
set -e
export PYTHONPATH=/workspace/artifacts/position:/workspace/artifacts/position/extension_aot_after:/workspace/benchmark
export POSITION_EXTENSION_NAME=position_aot_after POSITION_MAIN_MODULE=main_position
cd /workspace
python artifacts/position/run_accuracy.py
python benchmark/bench_position_occupancy.py --batch 8 --length-dtype int32 --prefix-mode zero --lengths 1024,1536,1792,4096,8193 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1
python artifacts/position/bench_split_vs_sycl.py
'

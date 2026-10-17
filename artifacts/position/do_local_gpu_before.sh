#!/usr/bin/env bash
set -euo pipefail
cd /home/syk/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014
trap 'printf "%s\n" "$?" > artifacts/position/local-int32-before.exit' EXIT
exec 9>/tmp/muse-gpu-bench.lock
flock 9
xpu-smi stats -d 0 > artifacts/position/local-int32-before-gpu-stats.log 2>&1
xpu-smi ps -d 0 > artifacts/position/local-int32-before-gpu-ps.log 2>&1
rg -q 'No data' artifacts/position/local-int32-before-gpu-ps.log
printf 'host=%s date=%s branch=%s\n' "$(hostname)" "$(date -Is)" "$(git rev-parse HEAD)"
SGLANG_MUSE_CONTAINER=codex-muse-position-local-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 0 --workdir /root/worktrees/sgl-kernel-xpu/muse-spark-position-main-20261014 -- bash -c '
set -e
export PYTHONPATH=/workspace/artifacts/position:/workspace/artifacts/position/extension_baseline:/workspace/benchmark
export POSITION_EXTENSION_NAME=position_baseline POSITION_MAIN_MODULE=main_position
cd /workspace
python benchmark/bench_position_occupancy.py --batch 8 --length-dtype int32 --prefix-mode zero --lengths 1,16,256,512,768,1024,1152,1280,1408,1536,1792,2048,4096,8193 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1
python benchmark/bench_position_occupancy.py --batch 8 --length-dtype int32 --prefix-mode zero --lengths 1,16,256,512,768,1024,1152,1280,1408,1536,1792,2048,4096,8193 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1 --reverse-order
python benchmark/bench_position_occupancy.py --batch 7 --length-dtype int32 --prefix-mode zero --lengths 4096 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1
python benchmark/bench_position_occupancy.py --batch 8 --length-dtype int32 --prefix-mode zero --lengths 4096 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1 --skew --max-seq-len-hint 4096
python benchmark/bench_position_occupancy.py --batch 8 --length-dtype int32 --prefix-mode zero --lengths 8193 --samples 31 --isl 4096 --osl 16 --tp 1 --ep 1 --skew --max-seq-len-hint 8193
python artifacts/position/bench_split_vs_sycl.py
POSITION_SPLIT_ORDER=split-first python artifacts/position/bench_main_split.py
POSITION_SPLIT_ORDER=main-first python artifacts/position/bench_main_split.py
'

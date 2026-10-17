#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if (($# != 1)); then
  echo 'ONE immutable new paired accuracy/trace grant required; no performance authorization'
  exit 2
fi
grant="$(realpath -- "$1")"
taskset -c 0-15 python3 - "$root" "$grant" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from contract import read,validate,unchanged,require,digest,local
from runner import live_builders
value=read(Path(sys.argv[2]))
validate(value,True)
unchanged()
require(digest(local(value['drain_path']))==value['drain_sha256'],'Fresh local prelaunch drain SHA')
require(read(local(value['drain_path']))['actual_exit']==0 and not live_builders(),'Compiler/package manager hold')
PY
flock -n /tmp/intel-gpu-6.lock true
mkdir "$root/host-run"
printf '%s\n' "$$" > "$root/host-run/launcher.pid"
container_root="${root/#\/home\/syk\//\/root\/}"
container_grant="${grant/#\/home\/syk\//\/root\/}"
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
command=(/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh --gpu-lock --gpu 6 --workdir "$container_root" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; exec taskset -c 32-47 python3 diagnose.py --mode suite --grant "$1"' bash "$container_grant")
printf '%q ' "${command[@]}" > "$root/host-run/command.txt"
printf '\n' >> "$root/host-run/command.txt"
set +e
"${command[@]}" > "$root/host-run/wrapper.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$root/host-run/wrapper.exit"
exit "$status"

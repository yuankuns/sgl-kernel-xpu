#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
if (($# != 1)); then
  echo 'Requires ONE new explicit external main grant; CPU-ready is NOT GPU authorization'
  exit 2
fi
grant="$(realpath -- "$1")"
read -r gpu cpus < <(taskset -c 0-15 python3 - "$grant" "$root" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from protocol import code_hashes,load_plan,validate_release
from runner import scope,live_builders,local
import time
scope()
value=json.loads(Path(sys.argv[1]).read_text())
plan,fields,cases=load_plan()
validate_release(value,plan,fields,cases,code_hashes(),time.time(),str(value.get('gpu')),value.get('runtime_cpus',[]),True)
evidence=local(value['drain_evidence_path'])
from protocol import digest,require
require(digest(Path(sys.argv[2])/'READY.json')==value.get('cpu_READY_sha256'),'Frozen v2 CPU READY required')
require(digest(evidence)==value['drain_evidence_sha256'],'New drain SHA required before wrapper/lock')
drain=json.loads(evidence.read_text())
require(drain.get('actual_audit_exit')==0 and drain.get('matching_live_compiler_processes')=={} and drain.get('matching_live_package_manager_processes')=={},'Compiler/package-manager hold before GPU wrapper')
require(not live_builders(),'Live compiler/package manager; no GPU lock/work')
print(value['gpu'],','.join(str(core) for core in value['runtime_cpus']))
PY
)
export SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015
container_root="${root/#\/home\/syk\//\/root\/}"
container_grant="${grant/#\/home\/syk\//\/root\/}"
wrapper=/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh
release_id="$(taskset -c 0-15 python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_id"])' "$grant")"
supervisor="$root/supervisors/$release_id"
mkdir -p "$root/supervisors"
mkdir "$supervisor"
printf '%s\n' "$$" > "$supervisor/host-supervisor.pid"
sha256sum "$grant" > "$supervisor/external-grant.sha256"
command=("$wrapper" --gpu-lock --gpu "$gpu" --workdir "$container_root" -- bash -c 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; export PYTHONDONTWRITEBYTECODE=1; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; exec taskset -c "$1" python3 runner.py --mode suite --grant "$2"' bash "$cpus" "$container_grant")
printf '%q ' "${command[@]}" > "$supervisor/command.txt"
printf '\n' >> "$supervisor/command.txt"
set +e
"${command[@]}" > "$supervisor/wrapper.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$supervisor/wrapper.exit"
exit "$status"

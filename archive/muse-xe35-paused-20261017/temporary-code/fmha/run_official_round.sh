#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
root=/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35
label=${1:?unique official round label required}
if [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then exit 2; fi
shift
python3 "$root/benchmark/bench_muse_fmha.py" --guard-only "$@"
for argument in "$@"; do
  case "$argument" in
    --serving-batch|--isl|--osl|--tp|--ep|--chunk|--warmup|--iterations|--replays) ;;
    --*) echo "round launcher rejects override $argument before GPU work" >&2; exit 2 ;;
  esac
done
python3 -S "$art/check_official.py" --preflight-only
test "$(cat "$art/canonical-v2/default-all-build.exit")" = 0
test "$(cat "$art/official-import.exit")" = 0
python3 - "$art" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

art = Path(sys.argv[1])
ready = json.loads((art / 'official-ready.json').read_text())
manifest = art / 'canonical-v2/source-binary-manifest.json'
assert ready['canonical_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
assert ready['diagnostic_AFTER_bridge'] is False
PY
exec 8>"$art/official-round.lock"
flock -n 8
python3 -u "$art/wait_gpu_barrier.py" --phase official
unset MUSE_SYNC_START_EPOCH
for slot in B0 A1 A2 B3; do
  side=port
  [[ $slot == B* ]] && side=source
  bash "$art/run_xe35.sh" "$side" "$label-r1-$slot" --mode bench "$@"
done
bash "$art/run_xe35.sh" source "$label-source-accuracy" --mode accuracy --reference full \
  --paired-dir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/paired-official-gpu5 --write-paired "$@"
bash "$art/run_xe35.sh" port "$label-port-accuracy" --mode accuracy --reference full \
  --paired-dir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/paired-official-gpu5 "$@"
python3 "$art/summarize.py" --prefix "$label" --official > "$art/$label-round1-summary.log"
focus=prefill-wave1-w-1,prefill-wave7-w-1,prefill-wave8-w-1,prefill-wave8-w2047,decode-osl1-w-1,decode-osl1-w2047,decode-osl16-w-1,decode-osl16-w2047
focus=$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(",".join(dict.fromkeys(sys.argv[2].split(",") + data["round_1"]["over_5_percent"])))' "$art/$label-summary.json" "$focus")
for slot in B0 A1 A2 B3; do
  side=port
  [[ $slot == B* ]] && side=source
  bash "$art/run_xe35.sh" "$side" "$label-r2-$slot" --mode bench --cases "$focus" "$@"
done
python3 "$art/summarize.py" --prefix "$label" --official > "$art/$label-summary.log"
echo 'GPU5 official round ended; all wrapper processes exited and GPU lock released'

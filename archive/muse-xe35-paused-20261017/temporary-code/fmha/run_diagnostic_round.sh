#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
root=/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35
label=${1:?unique diagnostic round label required}
if [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then exit 2; fi
shift
python3 "$root/benchmark/bench_muse_fmha.py" --guard-only "$@"
for argument in "$@"; do
  case "$argument" in
    --serving-batch|--isl|--osl|--tp|--ep|--chunk|--warmup|--iterations|--replays) ;;
    --*) echo "round launcher rejects override $argument before GPU work" >&2; exit 2 ;;
  esac
done
test -f "$art/diagnostic-provenance.json"
test "$(cat "$art/diagnostic-import.exit")" = 0
exec 8>"$art/diagnostic-round.lock"
flock -n 8
python3 -u "$art/wait_gpu_barrier.py" --phase diagnostic
bash "$art/run_xe35.sh" source "$label-r1-B0" --mode bench "$@"
unset MUSE_SYNC_START_EPOCH
bash "$art/run_xe35.sh" port-diagnostic "$label-r1-A1" --mode bench "$@"
bash "$art/run_xe35.sh" port-diagnostic "$label-r1-A2" --mode bench "$@"
bash "$art/run_xe35.sh" source "$label-r1-B3" --mode bench "$@"
python3 "$art/summarize.py" --prefix "$label" --diagnostic > "$art/$label-round1-summary.log"
bash "$art/run_xe35.sh" source "$label-source-accuracy" --mode accuracy --reference full \
  --paired-dir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/paired-gpu5 --write-paired "$@"
bash "$art/run_xe35.sh" port-diagnostic "$label-port-accuracy" --mode accuracy --reference full \
  --paired-dir /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha/paired-gpu5 "$@"
focus=prefill-wave1-w-1,prefill-wave7-w-1,prefill-wave8-w-1,prefill-wave8-w2047,decode-osl1-w-1,decode-osl1-w2047,decode-osl16-w-1,decode-osl16-w2047
focus=$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(",".join(dict.fromkeys(sys.argv[2].split(",") + data["round_1"]["over_5_percent"])))' "$art/$label-summary.json" "$focus")
bash "$art/run_xe35.sh" source "$label-r2-B0" --mode bench --cases "$focus" "$@"
bash "$art/run_xe35.sh" port-diagnostic "$label-r2-A1" --mode bench --cases "$focus" "$@"
bash "$art/run_xe35.sh" port-diagnostic "$label-r2-A2" --mode bench --cases "$focus" "$@"
bash "$art/run_xe35.sh" source "$label-r2-B3" --mode bench --cases "$focus" "$@"
python3 "$art/summarize.py" --prefix "$label" --diagnostic > "$art/$label-summary.log"
echo 'GPU5 diagnostic round ended; all wrapper processes exited and GPU lock released'

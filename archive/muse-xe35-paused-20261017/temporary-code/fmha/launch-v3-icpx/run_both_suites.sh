#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
launch="$art/launch-v3-icpx"
label=${1:?unique canonical-v3-icpx label required}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
shift
[[ $# == 0 ]] || { echo 'No workload/timing/reference overrides permitted' >&2; exit 2; }
python3 "$launch/bench_preopt_comparison.py" --guard-only
[[ ! -e "$art/$label-sourceport-suite.handle" && ! -e "$art/$label-preopt-suite.handle" ]] || exit 2
bash "$launch/run_official_import.sh"
bash "$art/supervise_xe35.sh" "$label-sourceport-suite" \
  bash "$launch/run_official_round.sh" "$label-sourceport"
bash "$art/supervise_xe35.sh" "$label-preopt-suite" \
  bash "$launch/run_preopt_round.sh" "$label-preopt"
echo 'Both icpx suites drained. GPU5 remains logically leased until main explicitly grants transfer.'

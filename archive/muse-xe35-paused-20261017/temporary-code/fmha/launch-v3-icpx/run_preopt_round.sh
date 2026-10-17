#!/usr/bin/env bash
set -euo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
version=$(python3 -S "$art/launch-v3-icpx/settings.py" --version-only)
label=${1:?unique preopt round label}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
shift
[[ $# == 0 ]] || { echo 'preopt suite accepts no scope/reference/timing overrides before GPU work' >&2; exit 2; }
python3 "$art/launch-v3-icpx/bench_preopt_comparison.py" --guard-only
python3 -S "$art/launch-v3-icpx/check_preopt.py"
python3 -S "$art/launch-v3-icpx/check_official.py" --preflight-only
test "$(cat "$art/official-import-$version.exit")" = 0
python3 - "$art" "$version" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

art = Path(sys.argv[1])
version = sys.argv[2]
ready = json.loads((art / version / 'normal-import-readiness.json').read_text())
assert ready['canonical_version'] == version
assert ready['canonical_manifest_sha256'] == hashlib.sha256((art / version / 'source-binary-manifest.json').read_bytes()).hexdigest()
assert ready['diagnostic_AFTER_bridge'] is False
PY
exec 8>"$art/official-round.lock"
flock -n 8
python3 -u "$art/launch-v3-icpx/wait_gpu_barrier.py" --phase official
for side in preopt port; do
  bash "$art/launch-v3-icpx/run_preopt_case.sh" "$side" "$label-$side-accuracy" --mode accuracy --reference full
done
python3 "$art/launch-v3-icpx/summarize_preopt.py" --prefix "$label" --accuracy-only > "$art/$label-accuracy-gate.log"
for slot in B0 A1 A2 B3; do
  side=port
  [[ $slot == B* ]] && side=preopt
  bash "$art/launch-v3-icpx/run_preopt_case.sh" "$side" "$label-r1-$slot" --mode bench
done
python3 "$art/launch-v3-icpx/summarize_preopt.py" --prefix "$label" > "$art/$label-round1-summary.log"
focus=prefill-wave1-w-1,prefill-wave7-w-1,prefill-wave8-w-1,prefill-wave8-w2047,decode-osl1-w-1,decode-osl1-w2047,decode-osl16-w-1,decode-osl16-w2047
focus=$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(",".join(dict.fromkeys(sys.argv[2].split(",") + data["round_1"]["over_5_percent"])))' "$art/$label-preopt-summary.json" "$focus")
for slot in B0 A1 A2 B3; do
  side=port
  [[ $slot == B* ]] && side=preopt
  bash "$art/launch-v3-icpx/run_preopt_case.sh" "$side" "$label-r2-$slot" --mode bench --cases "$focus"
done
python3 "$art/launch-v3-icpx/summarize_preopt.py" --prefix "$label" > "$art/$label-summary.log"
echo 'Preopt comparison drained; GPU5 logical lease retained until both suites finish and main grants transfer'

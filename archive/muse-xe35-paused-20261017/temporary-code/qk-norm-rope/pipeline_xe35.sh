#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$root/build-full_xe35.sh"
set +e
python3 "$root/prepare.py" > "$root/prepare.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/prepare.exit"
set -e
if ((status)); then
  tail -30 "$root/prepare.log"
  exit "$status"
fi
bash "$root/launch_xe35.sh"

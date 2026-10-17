#!/usr/bin/env bash
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$root/build-full-j8-v2_xe35.sh"
status=$?
if ((status == 0)); then
  python3 "$root/prepare.py" > "$root/prepare.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$root/prepare.exit"
fi
if ((status == 0)); then
  python3 "$root/audit_visibility.py" > "$root/visibility-audit.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$root/visibility-audit.exit"
fi
if ((status == 0)); then
  python3 "$root/report.py" > "$root/report.log" 2>&1
  printf '%s\n' "$?" > "$root/report.exit"
  bash "$root/launch_xe35.sh"
  status=$?
fi
printf '%s\n' "$status" > "$root/pipeline-full-j8-v2.exit"
python3 "$root/report.py" > "$root/report.log" 2>&1
printf '%s\n' "$?" > "$root/report.exit"
exit "$status"

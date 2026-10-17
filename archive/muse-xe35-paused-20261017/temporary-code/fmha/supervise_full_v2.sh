#!/usr/bin/env bash
set -uo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
name=full-build-v2-j8
test -f "$art/full-build-after-move.exit" || exit 125
test ! -f "$art/$name.handle" || exit 125
printf 'supervisor_pid=%s start_epoch=%s outer_jobs=8 inner_jobs=1 cpu_lane=1 command=' "$$" "$(date +%s)" > "$art/$name.handle"
printf '%q ' "$@" >> "$art/$name.handle"
printf '\n' >> "$art/$name.handle"
free -h > "$art/$name-resource-before.txt"
"$@" > "$art/$name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$art/$name.exit"
free -h > "$art/$name-resource-after.txt"
tail -c 7000 "$art/$name.log"
exit "$status"

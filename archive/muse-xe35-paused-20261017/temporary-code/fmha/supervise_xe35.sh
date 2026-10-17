#!/usr/bin/env bash
set -uo pipefail
art=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/fmha
name=$1
shift
printf 'supervisor_pid=%s command=' "$$" > "$art/$name.handle"
printf '%q ' "$@" >> "$art/$name.handle"
printf '\n' >> "$art/$name.handle"
"$@" > "$art/$name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$art/$name.exit"
tail -c 7000 "$art/$name.log"
exit "$status"

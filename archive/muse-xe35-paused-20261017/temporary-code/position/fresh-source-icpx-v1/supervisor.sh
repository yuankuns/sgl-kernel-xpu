#!/usr/bin/env bash
set -euo pipefail
root=/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
[[ "$#" == 2 ]] || { printf "%s\n" "usage: bash supervisor.sh UNIQUE_RUN MAIN_GRANT_JSON; builds all three SOURCE modules serially" >&2; exit 2; }
exec taskset -c 112-127 python3 -B "$root/memory/fresh-source-icpx-v1/sidecar.py" --run "$1" --grant "$2"

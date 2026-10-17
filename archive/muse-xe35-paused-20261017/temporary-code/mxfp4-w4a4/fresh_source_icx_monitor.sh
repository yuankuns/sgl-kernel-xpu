#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
exec "$artifact/port-venv/bin/python" /root/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35/tools/memory/compiler_launcher.py --db "$artifact/fresh-source-build/compile_mem.jsonl" --file-limit-gib 8 --file-limit-fatal /opt/gfx-deps/dpcpp/bin/icx "$@"

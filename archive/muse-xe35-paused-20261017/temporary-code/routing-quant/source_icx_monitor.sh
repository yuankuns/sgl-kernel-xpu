#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/routing-quant
exec "$artifact/port-venv/bin/python" /root/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35/tools/memory/compiler_launcher.py --db "$artifact/source-quant-build/compile_mem.jsonl" --file-limit-gib 8 --file-limit-fatal /opt/gfx-deps/dpcpp/bin/icx "$@"

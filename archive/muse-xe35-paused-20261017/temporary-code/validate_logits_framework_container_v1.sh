#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
artifact="$root/logits-vocab"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for variable in BATCHES BATCH_SIZE SERVING_BATCH; do
  if [[ "${!variable-8}" != 8 ]]; then exit 2; fi
done
export PYTHONPATH="$artifact/diagnostic/before/site:$artifact/framework-original/site:$root"
set +e
python - <<'PY' > "$artifact/diagnostic-framework-preflight-v1.log" 2>&1
import hashlib
import json
from pathlib import Path
import torch
import sglang
from sglang.srt.layers import vocab_parallel_embedding
assert callable(vocab_parallel_embedding.get_masked_input_and_mask)
filename = Path(vocab_parallel_embedding.__file__)
print(json.dumps({'torch': torch.__version__, 'sglang': sglang.__file__, 'framework_module': str(filename), 'sha256': hashlib.sha256(filename.read_bytes()).hexdigest()}))
PY
status=$?
set -e
printf '%s\n' "$status" > "$artifact/diagnostic-framework-preflight-v1.exit"
if (( status != 0 )); then tail -60 "$artifact/diagnostic-framework-preflight-v1.log"; exit "$status"; fi
cd "$artifact/native-diagnostic"
for label in before after; do
  export PYTHONPATH="$artifact/diagnostic/$label/site:$artifact/framework-original/site:$root"
  set +e
  python -m pytest -p acceptance_cooldown tests -q \
    > "$artifact/diagnostic-framework-v1-accuracy-$label.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/diagnostic-framework-v1-accuracy-$label.exit"
  if (( status != 0 )); then tail -80 "$artifact/diagnostic-framework-v1-accuracy-$label.log"; exit "$status"; fi
done
for run in B0 A0 A1 B1; do
  case "$run" in B*) label=before ;; A*) label=after ;; esac
  export PYTHONPATH="$artifact/diagnostic/$label/site:$artifact/framework-original/site"
  set +e
  python "$root/benchmark_simple.py" logits-vocab --api source-native --label "$label" \
    --output "$artifact/diagnostic-framework-v1-perf-$run.json" \
    > "$artifact/diagnostic-framework-v1-perf-$run.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$artifact/diagnostic-framework-v1-perf-$run.exit"
  if (( status != 0 )); then tail -60 "$artifact/diagnostic-framework-v1-perf-$run.log"; exit "$status"; fi
  sleep 2
done
python "$root/analyze_simple_diagnostic.py" logits-vocab --prefix diagnostic-framework-v1-perf \
  > "$artifact/diagnostic-framework-v1-perf-analysis.log" 2>&1

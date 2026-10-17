#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export MAX_JOBS=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for component in memory position logits-vocab; do
  artifact="$root/$component"
  build="$artifact/source-baseline-build"
  exec 8>"$artifact/build.lock"
  flock 8
  rm -f "$artifact/baseline-configure.exit" "$artifact/baseline-build.exit"
  set +e
  cmake -S "$artifact/source-baseline" -B "$build" -G Ninja \
    > "$artifact/baseline-configure.log" 2>&1
  configure_status=$?
  set -e
  printf '%s\n' "$configure_status" > "$artifact/baseline-configure.exit"
  if (( configure_status != 0 )); then tail -c 5000 "$artifact/baseline-configure.log"; exit "$configure_status"; fi
  set +e
  cmake --build "$build" --parallel 2 > "$artifact/baseline-build.log" 2>&1
  build_status=$?
  set -e
  printf '%s\n' "$build_status" > "$artifact/baseline-build.exit"
  if (( build_status != 0 )); then tail -c 5000 "$artifact/baseline-build.log"; exit "$build_status"; fi
  python - "$artifact" "$build" <<'PYTHON'
import hashlib
import json
import shutil
import sys
from pathlib import Path

artifact, build = map(Path, sys.argv[1:])
stage = artifact / 'before/site'
shutil.copytree(artifact / 'after/site', stage, dirs_exist_ok=True)
libraries = list(build.glob('libsgl-ops-sycl-*.so'))
if not libraries:
    raise RuntimeError('source baseline build did not produce native libraries')
for library in libraries:
    destination = stage / 'sgl_kernel' / library.name
    if not destination.is_file():
        raise RuntimeError(f'port package does not contain matching library: {destination}')
    shutil.copy2(library, destination)
manifest = {
    'baseline_method': 'unaltered original native kernels plus C++ ABI forwards; real port package registration',
    'baseline_build': str(build),
    'files': {
        str(filename.relative_to(artifact)): hashlib.sha256(filename.read_bytes()).hexdigest()
        for folder in (artifact / 'source-baseline', stage)
        for filename in folder.rglob('*') if filename.is_file()
    },
}
(artifact / 'baseline-source-binary-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PYTHON
  flock -u 8
done

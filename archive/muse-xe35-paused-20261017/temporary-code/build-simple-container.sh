#!/usr/bin/env bash
set -euo pipefail
umask 000
root=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
export MAX_JOBS=1
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
for component in memory position logits-vocab; do
  source="/root/worktrees/sgl-kernel-xpu-up/pr/muse-${component}-xe35"
  artifact="$root/$component"
  build="$artifact/full-default-build"
  mkdir -p "$artifact"
  exec 8>"$artifact/build.lock"
  flock 8
  rm -f "$artifact/configure.exit" "$artifact/full-default-build.exit"
  printf '%s\n' "cmake -S $source -B $build -G Ninja -DDPCPP_SYCL_TARGET=cri -DCMAKE_BUILD_TYPE=Release -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.10 -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=$root/deps/sycl-tla-7b56 -DSGL_BUILD_MEM_FILE_LIMIT_GIB=8" > "$artifact/configure.command"
  set +e
  cmake -S "$source" -B "$build" -G Ninja \
    -DDPCPP_SYCL_TARGET=cri \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON3_EXECUTABLE=/opt/gfx-deps/venv/bin/python \
    -DSKBUILD_SABI_COMPONENT=Development.SABIModule \
    -DSKBUILD_SABI_VERSION=3.10 \
    -DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL="$root/deps/sycl-tla-7b56" \
    -DSGL_BUILD_MEM_FILE_LIMIT_GIB=8 > "$artifact/configure.log" 2>&1
  configure_status=$?
  set -e
  printf '%s\n' "$configure_status" > "$artifact/configure.exit"
  if (( configure_status != 0 )); then tail -50 "$artifact/configure.log"; exit "$configure_status"; fi
  printf '%s\n' "cmake --build $build --parallel 8" > "$artifact/full-default-build.command"
  set +e
  cmake --build "$build" --parallel 8 > "$artifact/full-default-build.log" 2>&1
  build_status=$?
  set -e
  printf '%s\n' "$build_status" > "$artifact/full-default-build.exit"
  tail -15 "$artifact/full-default-build.log"
  if (( build_status != 0 )); then exit "$build_status"; fi
  python - "$source" "$build" "$artifact" <<'PYTHON'
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

source, build, artifact = map(Path, sys.argv[1:])
stage = artifact / 'after/site'
stage.mkdir(parents=True, exist_ok=True)
shutil.copytree(source / 'python/sgl_kernel', stage / 'sgl_kernel', dirs_exist_ok=True)
for library in (build / 'src').glob('*.so'):
    shutil.copy2(library, stage / 'sgl_kernel' / library.name)
for library in (build / 'src').rglob('common_ops*.so'):
    shutil.copy2(library, stage / 'sgl_kernel' / library.name)
files = subprocess.check_output(
    ['git', '-C', str(source), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
).decode().split('\0')
manifest = {
    'head': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
    'source': str(source),
    'files': {
        filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
        for filename in files if filename and (source / filename).is_file()
    },
    'stage': {
        str(filename.relative_to(stage)): hashlib.sha256(filename.read_bytes()).hexdigest()
        for filename in stage.rglob('*') if filename.is_file()
    },
}
(artifact / 'source-binary-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PYTHON
  flock -u 8
done

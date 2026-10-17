#!/usr/bin/env bash
set -euo pipefail
artifact=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4
[[ "$(cat "$artifact/fresh-source-build.exit")" == 0 ]] || exit 125
export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
"$artifact/source-venv/bin/python" "$artifact/package_fresh_source.py"
[[ -d "$artifact/fresh-source-venv" ]] || cp -a "$artifact/source-venv" "$artifact/fresh-source-venv"
"$artifact/fresh-source-venv/bin/python" -m pip install --force-reinstall --no-deps "$artifact"/fresh-source-wheels/*.whl
"$artifact/fresh-source-venv/bin/python" - <<'PY'
import hashlib
import json
import pathlib
import zipfile
import sgl_kernel
artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/mxfp4-w4a4')
package = pathlib.Path(sgl_kernel.__file__).parent
provenance = json.loads((artifact / 'fresh-source-package-provenance.json').read_text())
native = package / 'libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so'
actual = hashlib.sha256(native.read_bytes()).hexdigest()
assert actual == provenance['fresh_native_sha256']
assert str(native) in pathlib.Path('/proc/self/maps').read_text()
wheel = next((artifact / 'source-wheels').glob('*.whl'))
with zipfile.ZipFile(wheel) as source:
    for name in source.namelist():
        if name.endswith('.py') or '/common_ops.' in name:
            assert (package / pathlib.Path(name).relative_to('sgl_kernel')).read_bytes() == source.read(name), name
record = {'package': str(package), 'fresh_original_native_sha256': actual, 'fresh_original_native_actually_loaded': True, 'original_python_and_common_ops': 'byte-identical', 'acceptance': 'CPU import provenance only; diagnostic not official full-package pass', 'gpu_initialized': False}
(artifact / 'fresh-source-import-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY

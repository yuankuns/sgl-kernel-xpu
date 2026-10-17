#!/usr/bin/env bash
set -euo pipefail
module="${1:?module required}"
case "$module" in mxfp4-w4a4|routing-quant) ;; *) exit 2 ;; esac
base=/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017
artifact="$base/$module"
export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export http_proxy=http://child-prc.intel.com:913 https_proxy=http://child-prc.intel.com:913
export PATH="$artifact/port-venv/bin:$PATH"
if ! command -v patchelf >/dev/null; then python -m pip install --only-binary=:all: patchelf; fi
python "$base/mxfp4-w4a4/package_diagnostic.py" --module "$module"
if [[ ! -d "$artifact/diagnostic-venv" ]]; then cp -a "$artifact/source-venv" "$artifact/diagnostic-venv"; fi
"$artifact/diagnostic-venv/bin/python" -m pip install --force-reinstall --no-deps "$artifact"/diagnostic-wheels/*.whl
"$artifact/diagnostic-venv/bin/python" - "$module" "$artifact" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile

import sgl_kernel

module, artifact_name = sys.argv[1:]
artifact = pathlib.Path(artifact_name)
package_root = pathlib.Path(sgl_kernel.__file__).parent
provenance = json.loads((artifact / "diagnostic-package-provenance.json").read_text())
maps = pathlib.Path("/proc/self/maps").read_text()
for name, expected in provenance["relocated_native_hashes"].items():
    path = package_root / name
    assert str(path) in maps, f"actual native library not loaded: {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
with zipfile.ZipFile(next((artifact / "source-wheels").glob("*.whl"))) as source:
    for name in source.namelist():
        if name.endswith(".py") or pathlib.Path(name).name.startswith("common_ops"):
            actual = package_root / pathlib.Path(name).relative_to("sgl_kernel")
            assert actual.read_bytes() == source.read(name), name
report = {"package": str(package_root), "module": module, "kind": "real port native via byte-identical original complete source Python and registration, NOT official full-package acceptance", "loaded_port_libraries": list(provenance["relocated_native_hashes"])}
(artifact / "diagnostic-import-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
PY
sha256sum "$artifact"/diagnostic-wheels/*.whl

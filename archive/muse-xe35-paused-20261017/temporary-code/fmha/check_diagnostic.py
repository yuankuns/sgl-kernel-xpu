import hashlib
import json
from pathlib import Path

import torch
import sgl_kernel

art = Path(__file__).resolve().parent
site = art / "site-port-diagnostic/sgl_kernel"
assert Path(sgl_kernel.__file__).resolve().parent == site
assert not torch.xpu.is_initialized(), "readiness import must not initialize GPU"
with open("/proc/self/maps") as maps:
    loaded = {line.split()[-1] for line in maps if "/sgl_kernel/" in line and ".so" in line}
providers = {"libsgl-ops-sycl-fmha_decode_64_bf16.so", "libsgl-ops-sycl-fmha_prefill_64_bf16.so"}
assert providers <= {Path(path).name for path in loaded}
assert all(Path(path).parent == site for path in loaded), loaded
report = {"ready": True, "acceptance": "diagnostic_import_only_not_gpu_or_official_full_pass",
          "gpu_lease": 5, "cpu_lane": 1, "gpu_initialized": torch.xpu.is_initialized(),
          "package": str(sgl_kernel.__file__),
          "libraries": [{"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                        for path in sorted(loaded)]}
(art / "diagnostic-ready.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

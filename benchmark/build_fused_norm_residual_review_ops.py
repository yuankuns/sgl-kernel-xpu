"""Build only the registration needed for targeted residual-norm source tests."""

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "review_logs" / "ops"
BUILD.mkdir(parents=True, exist_ok=True)
os.environ["MAX_JOBS"] = "1"

load(
    name="sgl_kernel_review_ops",
    sources=[str(ROOT / "benchmark" / "fused_norm_residual_review_ops.cpp")],
    extra_include_paths=[str(ROOT / "include")],
    extra_ldflags=[
        str(ROOT / "build" / "src" / "libsgl-ops-sycl-FusedNormResidual.so"),
        str(ROOT / "build" / "src" / "libsgl-ops-sycl-RMSNorm.so"),
        f"-Wl,-rpath,{ROOT / 'build' / 'src'}",
    ],
    build_directory=str(BUILD),
    is_python_module=False,
    verbose=True,
)

site = ROOT / "review_logs" / "local_site" / "sgl_kernel"
site.mkdir(parents=True, exist_ok=True)
(site / "__init__.py").write_text(
    "import torch\n"
    f"torch.ops.load_library({str(BUILD / 'sgl_kernel_review_ops.so')!r})\n"
)

import hashlib
import pathlib
import subprocess

import torch
from setuptools import Extension, setup
from torch.utils.cpp_extension import include_paths, library_paths

source = pathlib.Path("/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014")
revision = subprocess.check_output(["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert revision == "d983e877d6bf97015027bd3b1379ff047009a5dd", revision
libraries = {
    "libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so": "864f695ee11647af36ea1140c941d3ce6f68ae3c1c22a5928784d34766ff4e06",
    "libsgl-ops-sycl-nvfp4_woq.so": "bad2595a97d75d903fecac9770ddc2bf659baaee1cff362c84204108b6c131fa",
    "libsgl-ops-sycl-mxfp4_moe_scale_layout.so": "2196b48e0e601bd78f9610bc2b6f5541c8942ba7b5e78d30af2946acef33e3d4",
}
objects = []
for name, expected in libraries.items():
    path = source / "build/src" / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (path, actual, expected)
    objects.append(str(path))

setup(ext_modules=[Extension(
    "source_nvfp4_woq_xe35",
    sources=["source_bridge_xe35.cpp"],
    include_dirs=include_paths(),
    library_dirs=library_paths() + ["/opt/gfx-deps/dpcpp/lib"],
    libraries=["torch_python", "torch_xpu", "torch_cpu", "torch", "c10_xpu", "c10", "sycl"],
    extra_objects=objects,
    extra_compile_args=["-O2", "-std=c++20", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"],
    extra_link_args=["-Wl,--no-as-needed"],
    runtime_library_dirs=[str(source / "build/src")] + library_paths() + ["/opt/gfx-deps/dpcpp/lib"],
    language="c++",
)])

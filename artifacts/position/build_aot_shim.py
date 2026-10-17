import os
from torch.utils.cpp_extension import load

os.environ["MAX_JOBS"] = "1"
load(
    name="position_aot_after",
    sources=["/workspace/artifacts/position/position_register.cc"],
    extra_cflags=["-O2"],
    extra_ldflags=[
        "/workspace/build/src/libsgl-ops-sycl-ComputePosition.so",
        "-Wl,-rpath,/workspace/build/src",
    ],
    build_directory="/workspace/artifacts/position/extension_aot_after",
    verbose=True,
)

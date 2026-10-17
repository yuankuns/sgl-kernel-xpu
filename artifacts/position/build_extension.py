import os
from torch.utils.cpp_extension import load

os.environ["MAX_JOBS"] = "1"
load(
    name=os.environ.get("POSITION_EXTENSION_NAME", "position_baseline"),
    sources=[os.environ.get("POSITION_SOURCE", "/workspace/artifacts/position/MuseComputePosition.before.cpp"), "/workspace/artifacts/position/position_register.cc"],
    extra_include_paths=["/workspace/src/sycl"],
    extra_cflags=["-O2", "-fsycl"],
    extra_ldflags=["-fsycl"],
    build_directory=os.environ.get("POSITION_BUILD_DIRECTORY", "/workspace/artifacts/position/extension_baseline"),
    verbose=True,
)

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).resolve().parent
setup(
    name="muse-residual-norm-source-xe35",
    version="0.0.1",
    packages=["muse_residual_norm_source_xe35"],
    package_data={"muse_residual_norm_source_xe35": ["*.so"]},
    ext_modules=[CppExtension(
        "muse_residual_norm_source_xe35.source_ops_xe35",
        [str(root / "source_ops_xe35.cpp")],
        extra_compile_args=["-std=c++20"],
        extra_link_args=[
            str(root / "muse_residual_norm_source_xe35/libsgl-ops-sycl-FusedNormResidual.so"),
            "-Wl,-rpath,$ORIGIN",
        ],
    )],
    cmdclass={"build_ext": BuildExtension},
)

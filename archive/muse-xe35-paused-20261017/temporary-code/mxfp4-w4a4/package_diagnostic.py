import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sysconfig
import zipfile

import torch
from wheel.wheelfile import WheelFile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=("mxfp4-w4a4", "routing-quant"), required=True)
    args = parser.parse_args()
    base = pathlib.Path("/root/worktrees/sgl-kernel-xpu-up")
    artifact = base / "muse-upstream-artifacts-20261017" / args.module
    target = base / "pr" / ("muse-" + args.module + "-xe35")
    source = pathlib.Path("/root/worktrees/sgl-kernel-xpu") / (
        "muse-spark-mxfp4-" + ("w4a4" if args.module == "mxfp4-w4a4" else "routing-quant") + "-main-20261014"
    )
    source_wheel = next((artifact / "source-wheels").glob("*.whl"))
    source_provenance = json.loads((artifact / "source-package-provenance.json").read_text())
    if digest(source_wheel) != source_provenance["wheel_sha256"]:
        raise RuntimeError("original complete source package hash mismatch")
    destination = artifact / "diagnostic-native"
    destination.mkdir(exist_ok=True)
    if args.module == "mxfp4-w4a4":
        native_names = ["moe_grouped_mm_mxfp4_w4a4", "mxfp4_moe_scale_layout"]
        old_names = native_names
    else:
        native_names = ["muse_routing_quant", "per_token_group_quant_fp4_xe35", "mxfp4_moe_scale_layout"]
        old_names = ["per_token_group_quant_fp4", "mxfp4_moe_scale_layout"]
    copied = []
    native_hashes = {}
    for name in native_names:
        original = target / "build/src" / ("libsgl-ops-sycl-" + name + ".so")
        copy = destination / ("libmuse-port-" + args.module + "-" + name + ".so")
        shutil.copy2(original, copy)
        subprocess.run(["patchelf", "--set-soname", copy.name, str(copy)], check=True)
        copied.append(copy)
        native_hashes[str(original)] = digest(original)
    includes = [source / "include", target / "include", pathlib.Path(torch.__file__).parent / "include", pathlib.Path(torch.__file__).parent / "include/torch/csrc/api/include", pathlib.Path(sysconfig.get_paths()["include"]), pathlib.Path("/opt/gfx-deps/dpcpp/include")]
    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    bridge = destination / "libmuse-source-abi-forwards.so"
    command = ["g++", "-std=c++20", "-O3", "-fPIC", "-shared", "-DSYCL_INTEL_TARGET=35", "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    command += ["-I" + str(path) for path in includes]
    command += [str(artifact / "abi_forwards.cpp")]
    command += [str(path) for path in copied]
    if args.module == "routing-quant":
        fallback = destination / "libmuse-source-topk-fallback.so"
        with zipfile.ZipFile(source_wheel) as package:
            fallback.write_bytes(package.read("sgl_kernel/libsgl-ops-sycl-TopKSigMoid.so"))
        subprocess.run(["patchelf", "--set-soname", fallback.name, str(fallback)], check=True)
        command += [str(fallback)]
        copied.append(fallback)
    command += ["-L" + str(torch_lib), "-ltorch", "-ltorch_cpu", "-lc10", "-Wl,--no-undefined", "-Wl,-rpath,$ORIGIN", "-Wl,-rpath," + str(torch_lib), "-o", str(bridge)]
    (artifact / "diagnostic-forward-compile.command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    replaced = {"libsgl-ops-sycl-" + name + ".so" for name in old_names}
    if args.module == "routing-quant":
        replaced.add("libsgl-ops-sycl-TopKSigMoid.so")
    wheel = artifact / "diagnostic-wheels" / source_wheel.name
    wheel.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_wheel) as original, WheelFile(wheel, "w") as package:
        for member in original.namelist():
            if member.endswith("/RECORD") or pathlib.Path(member).name in replaced:
                continue
            package.writestr(member, original.read(member))
        for path in copied:
            package.write(path, "sgl_kernel/" + path.name)
        for name in sorted(replaced):
            package.write(bridge, "sgl_kernel/" + name)
    report = {
        "kind": "actual port native kernels through complete original source package and pure C++ ABI forwards; NOT official full-package acceptance",
        "source_wheel_sha256": digest(source_wheel),
        "diagnostic_wheel_sha256": digest(wheel),
        "port_native_hashes": native_hashes,
        "relocated_native_hashes": {path.name: digest(path) for path in copied},
        "abi_forward_source_sha256": digest(artifact / "abi_forwards.cpp"),
        "abi_forward_binary_sha256": digest(bridge),
        "registration": "original complete common_ops and original Python retained byte-for-byte; no fake registration, shim Python, or load_library",
        "metadata_only_native_relocation": "copied ELF SONAME changed to prevent alias collision; kernel code not modified",
    }
    (artifact / "diagnostic-package-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

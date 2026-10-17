import hashlib
import json
import pathlib
import shutil
import subprocess
import sysconfig

import torch
from wheel.wheelfile import WheelFile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    artifact = pathlib.Path(__file__).resolve().parent
    source = pathlib.Path("/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014")
    baseline = pathlib.Path("/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/muse_mxfp4_w4a4")
    reviewed = pathlib.Path("/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_routing_quant")
    destination = artifact / "source-native"
    destination.mkdir(exist_ok=True)
    checked = {}
    modified = {"src/sycl/TopKSigMoid.cpp", "src/sycl/TripleOps.cpp", "src/sycl/per_token_group_quant_fp4.cpp", "src/sycl/mxfp4_moe_scale_layout.cpp", "src/torch_extension_sycl.cc", "include/sgl_kernel_ops.h", "src/CMakeLists.txt", "src/BuildOnLinux.cmake"}
    previous = json.loads((baseline / "full-default-source.json").read_text())["files"]
    for name, expected in previous.items():
        path = source / name
        if name.startswith(("src/", "include/")) and path.exists() and name not in modified:
            actual = digest(path)
            if actual != expected:
                raise RuntimeError(f"generic source reuse hash mismatch: {name}")
            checked[name] = actual
    frozen = {entry["path"]: entry["sha256"] for entry in json.loads((reviewed / "frozen-source.json").read_text())["files"]}
    for name in modified:
        if name == "src/sycl/TripleOps.cpp":
            expected = previous[name]
        elif name == "src/sycl/per_token_group_quant_fp4.cpp":
            expected = "5a9088e18956818b9d295c0deedf9c57998c6ea5a8a655560a64dba0973646d2"
        elif name == "src/BuildOnLinux.cmake":
            expected = "e91c8a092bc14a5dbcbce86b22bb3f1f0eaa5186a0f25673e3d0e30765a53f3a"
        else:
            expected = frozen[name]
        actual = digest(source / name)
        if actual != expected:
            raise RuntimeError(f"reviewed source hash mismatch: {name}")
        checked[name] = actual
    binaries = {}
    for line in (baseline / "full-default-binaries.sha256").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = pathlib.Path(filename)
        if path.name.startswith("common_ops") or "moe_grouped_mm_mxfp4_w4a4" in path.name:
            continue
        if digest(path) != expected:
            raise RuntimeError(f"generic binary reuse hash mismatch: {path}")
        binaries[path.name] = path
    for line in (reviewed / "after-built-sha256.txt").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = pathlib.Path(filename)
        if path.name.startswith("common_ops") or path.name == "libsgl-ops-sycl-per_token_group_quant_fp4.so":
            continue
        if digest(path) != expected:
            raise RuntimeError(f"reviewed binary reuse hash mismatch: {path}")
        binaries[path.name] = path
    fresh_quant = artifact / "source-quant-build/src/libsgl-ops-sycl-per_token_group_quant_fp4.so"
    if not fresh_quant.exists():
        raise RuntimeError("current source quant precision requires a fresh native build; historical quant hash is obsolete")
    binaries[fresh_quant.name] = fresh_quant
    for name, path in binaries.items():
        shutil.copy2(path, destination / name)
    includes = [source / "include", source / "src", pathlib.Path(torch.__file__).parent / "include", pathlib.Path(torch.__file__).parent / "include/torch/csrc/api/include", pathlib.Path(sysconfig.get_paths()["include"]), pathlib.Path("/opt/gfx-deps/dpcpp/include")]
    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    command = ["g++", "-O3", "-std=c++20", "-fPIC", "-shared", "-DSYCL_INTEL_TARGET=35", "-DPy_LIMITED_API=0x03090000", "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    command += ["-I" + str(path) for path in includes]
    command += [str(source / "src/torch_extension_sycl.cc"), str(source / "src/weak_ref_tensor.cpp")]
    command += [str(destination / name) for name in sorted(binaries)]
    command += ["-L" + str(torch_lib), "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10", "-Wl,-rpath,$ORIGIN", "-Wl,-rpath," + str(torch_lib), "-o", str(destination / "common_ops.abi3.so")]
    (artifact / "source-common-compile.command.json").write_text(json.dumps(command, indent=2) + "\n")
    monitor = [str(artifact / "port-venv/bin/python"), str(pathlib.Path("/root/worktrees/sgl-kernel-xpu-up/pr/muse-routing-quant-xe35/tools/memory/compiler_launcher.py")), "--db", str(artifact / "source-native/compile_mem.jsonl"), "--file-limit-gib", "8", "--file-limit-fatal"]
    subprocess.run(monitor + command, check=True)
    binaries["common_ops.abi3.so"] = destination / "common_ops.abi3.so"
    wheel = artifact / "source-wheels/sgl_kernel-0.11.0-cp39-abi3-linux_x86_64.whl"
    wheel.parent.mkdir(exist_ok=True)
    python_hashes = {}
    with WheelFile(wheel, "w") as package:
        for path in sorted((source / "python/sgl_kernel").rglob("*.py")):
            name = str(path.relative_to(source / "python"))
            package.write(path, name)
            python_hashes[name] = digest(path)
        for name in sorted(binaries):
            package.write(destination / name, "sgl_kernel/" + name)
        metadata = "sgl_kernel-0.11.0.dist-info/"
        package.writestr(metadata + "METADATA", "Metadata-Version: 2.1\nName: sgl-kernel\nVersion: 0.11.0\n")
        package.writestr(metadata + "WHEEL", "Wheel-Version: 1.0\nGenerator: verified-source-native-reuse\nRoot-Is-Purelib: false\nTag: cp39-abi3-linux_x86_64\n")
    report = {"source_revision": "4cb5e92", "source_native_hashes": checked, "source_python_hashes": python_hashes, "binary_hashes": {name: digest(destination / name) for name in binaries}, "wheel_sha256": digest(wheel), "registration": "fresh compile of the complete original torch_extension_sycl.cc and weak_ref_tensor.cpp, no scoped shim", "reuse": "hash-verified unchanged generic/routing/relayout binaries; current original quant fresh-built because archived quant source hash differs; no claim of fresh source full build", "build_script_difference": "W4 baseline has only an added 512-GRF flag for its W4A4 native target, which is excluded from routing reuse; generic compile flags otherwise identical", "resources": "lane4 CPU64-79 jobs2 cap8; fresh compiler wrappers enforce per-file 8GiB, not feature disable"}
    (artifact / "source-package-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wheel": str(wheel), "sha256": report["wheel_sha256"]}), flush=True)


if __name__ == "__main__":
    main()

import hashlib
import json
import pathlib
import subprocess

from wheel.wheelfile import WheelFile


def digest_xe35(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main_xe35():
    source = pathlib.Path("/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014")
    historical = pathlib.Path("/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/muse_mxfp4_w4a4")
    artifact = pathlib.Path(__file__).resolve().parent
    source_files = json.loads((historical / "full-default-source.json").read_text())["files"]
    checked_sources = {}
    for name, expected in source_files.items():
        if name == "CMakeLists.txt" or name.startswith(("src/", "include/", "cmake/")):
            path = source / name
            if name == "src/sycl/TripleOps.cpp":
                expected = (historical / "tripleops-before.sha256").read_text().split()[0]
            actual = digest_xe35(path)
            if actual != expected:
                raise RuntimeError(f"source hash mismatch: {name}")
            checked_sources[name] = actual
    binaries = {}
    for line in (historical / "full-default-binaries.sha256").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = pathlib.Path(filename)
        if digest_xe35(path) != expected:
            raise RuntimeError(f"binary hash mismatch: {path}")
        binaries[path.name] = path
    expected = (historical / "activation-baseline.sha256").read_text().split()[0]
    restored = historical / "activation-baseline-built/libsgl-ops-sycl-TripleOps.so"
    if digest_xe35(restored) != expected:
        raise RuntimeError("restored unrelated activation binary hash mismatch")
    binaries[restored.name] = restored
    wheel = artifact / "source-wheels/sgl_kernel-0.11.0-cp39-abi3-linux_x86_64.whl"
    wheel.parent.mkdir(exist_ok=True)
    python_hashes = {}
    with WheelFile(wheel, "w") as package:
        for path in sorted((source / "python/sgl_kernel").rglob("*.py")):
            name = str(path.relative_to(source / "python"))
            package.write(path, name)
            python_hashes[name] = digest_xe35(path)
        for name, path in sorted(binaries.items()):
            package.write(path, "sgl_kernel/" + name)
        metadata = "sgl_kernel-0.11.0.dist-info/"
        package.writestr(metadata + "METADATA", "Metadata-Version: 2.1\nName: sgl-kernel\nVersion: 0.11.0\n")
        package.writestr(metadata + "WHEEL", "Wheel-Version: 1.0\nGenerator: verified-source-artifact-reuse-xe35\nRoot-Is-Purelib: false\nTag: cp39-abi3-linux_x86_64\n")
    git_dir = (source / ".git").read_text().strip().removeprefix("gitdir: ").replace("/home/syk/", "/root/")
    provenance = {
        "source": str(source),
        "source_head": subprocess.check_output(["git", "--git-dir=" + git_dir, "rev-parse", "HEAD"], text=True).strip(),
        "source_native_hashes": checked_sources,
        "source_python_hashes": python_hashes,
        "reused_binary_hashes": {name: {"path": str(path), "sha256": digest_xe35(path)} for name, path in binaries.items()},
        "wheel_sha256": digest_xe35(wheel),
        "limits": "历史全量 native 构建复用；TripleOps 使用已恢复且验 hash 的 baseline binary。不是新鲜 source build，不将历史测试当本轮通过。",
    }
    (artifact / "source-package-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(wheel, provenance["wheel_sha256"], flush=True)


if __name__ == "__main__":
    main_xe35()

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.component in {"bf16-gemm", "fmha", "qk-norm-rope", "mxfp4-w4a4", "residual-norm", "nvfp4-woq", "moe-permutation", "routing-quant", "memory", "position", "logits-vocab"}
    stage = args.root / args.component / "canonical-v4-icpx"
    manifest_path = stage / "source-binary-manifest.json"
    manifest = load(manifest_path)
    frozen = load(args.root / args.component / "canonical-source-freeze-v4-icpx.json")
    policy = load(stage / "compiler-policy-verified.json")
    assert policy["compiler_id"] == "IntelLLVM" and policy["no_GNU_command"] is True
    for key in ["cxx", "sycl", "host"]:
        assert policy[key] == "/opt/gfx-deps/dpcpp/bin/icpx"
    assert policy["c"] == "/opt/gfx-deps/dpcpp/bin/icx"
    assert manifest["source"]["source_digest"] == frozen["source_digest"]
    assert load(stage / "source-readonly-postverify.json")["verified"] is True
    for phase in ["configure", "default-all-build", "supervisor"]:
        assert (stage / f"{phase}.exit").read_text().strip() == "0"
    assert manifest["install_actual_exit"] == 0
    evidence = manifest["build_evidence"]
    assert evidence["exit"] == 0
    assert evidence["command_text"].strip() == f'cmake --build {manifest["build"]} --parallel 56'
    for filename_key, sha_key in [("command_file", "command_sha256"), ("exit_file", "exit_file_sha256"), ("log_file", "log_sha256")]:
        assert digest(Path(evidence[filename_key])) == evidence[sha_key]
    assert digest(stage / "CMakeCache.txt") == manifest["cache_sha256"]
    assert digest(stage / "build.ninja") == manifest["ninja_sha256"]
    expected_files = manifest["installed_files"] | manifest["python_snapshot_files"]
    site = Path(manifest["site"])
    for relative, record in expected_files.items():
        assert digest(site / relative) == record["sha256"], relative
    for record in manifest["binaries"].values():
        assert digest(site / record["path"]) == record["sha256"]
    assert manifest["cutlass_pin"] == "7b56b936b87f799f80dde60c2fd8d11c1128cebb"
    assert manifest["acceptance_complete"] is False
    report = {
        "component": args.component,
        "version": "canonical-v4-icpx",
        "epoch": time.time(),
        "source_digest": frozen["source_digest"],
        "package_manifest_sha256": digest(manifest_path),
        "compiler_policy_sha256": digest(stage / "compiler-policy-verified.json"),
        "actual_cxx_sycl_host_compiler": "icpx",
        "default_all_actual_exit": 0,
        "install_actual_exit": 0,
        "source_postverified": True,
        "installed_and_python_files_rehashed": len(expected_files),
        "native_binary_inventory_rehashed": len(manifest["binaries"]),
        "build_scope": "fresh default-all graph; initial family clean, later families real graph incremental; no copied native objects",
        "verified": True,
        "normal_package_runtime_accuracy_performance_verified": False,
        "GPU_started": False,
    }
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

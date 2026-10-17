import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def git(source, *arguments):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), *arguments]
    )


def audit_source(root, component):
    prepared = root / component / "fresh-source-icpx-v1"
    attempt = prepared / "runs/source-native-icpx-20261017-r1"
    inputs = load(prepared / "inputs.json")
    manifest = load(attempt / "tested-native-source-manifest.json")
    snapshot = load(attempt / "source-snapshot.json")
    policy = load(attempt / "compiler-policy-verified.json")
    source = Path(inputs["source_worktree"].replace("/home/syk/", "/root/", 1))
    archive_bytes = git(source, "archive", "--format=tar", manifest["source_revision"])
    assert hashlib.sha256(archive_bytes).hexdigest() == snapshot["archive_sha256"]
    assert digest(attempt / "original-source.tar") == snapshot["archive_sha256"]
    assert git(source, "rev-parse", f'{manifest["source_revision"]}^{{tree}}').decode().strip() == snapshot["original_tree"]
    checked = 0
    with tarfile.open(attempt / "original-source.tar") as archive:
        for member in archive.getmembers():
            target = attempt / "source" / member.name
            if member.isfile():
                expected = archive.extractfile(member).read()
                if member.name == "cmake/BuildFlags.cmake":
                    expected = expected.replace(
                        b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")',
                        b'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR\n   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")',
                    )
                assert target.read_bytes() == expected, (component, member.name)
                checked += 1
            elif member.issym():
                assert target.is_symlink() and target.readlink() == Path(member.linkname)
    assert checked == snapshot["files_checked"]
    assert digest(prepared / "inputs.json") == manifest["source_input_manifest_sha256"]
    assert digest(attempt / "compiler-policy-verified.json") == manifest["compiler_policy_sha256"]
    assert policy["compiler_id"] == "IntelLLVM"
    assert policy["source_flags_AOT_preserved"] is True
    for key in ["CMAKE_CXX_COMPILER", "SYCL_COMPILER", "SYCL_HOST_COMPILER"]:
        assert policy[key] == "/opt/gfx-deps/dpcpp/bin/icpx"
    assert policy["CMAKE_C_COMPILER"] == "/opt/gfx-deps/dpcpp/bin/icx"
    phases = ["torch-cpu-probe", "configure", "tested-native-build", "supervisor"]
    for phase in phases:
        assert (attempt / f"{phase}.exit").read_text().strip() == "0"
        terminal = load(attempt / f"{phase}.terminal.json")
        drain = load(attempt / f"{phase}.drain.json")
        assert terminal["actual_exit"] == 0 and terminal["all_descendants_drained"] is True
        assert drain["drained"] is True and drain["live_after"] == []
        for identity in drain["identities"]:
            stat = Path(f'/proc/{identity["pid"]}/stat')
            if stat.exists():
                fields = stat.read_text().rsplit(")", 1)[1].split()
                assert fields[0] == "Z" or int(fields[19]) != identity["start_ticks"]
    native = {}
    for name, record in manifest["native"].items():
        assert digest(attempt / "build" / name) == record["sha256"]
        native[name] = record["sha256"]
    samples = [load_line for line in (attempt / "tested-native-build.resources.jsonl").read_text().splitlines() if (load_line := json.loads(line))]
    peak_tree = max(sample["tree_rss_gib"] for sample in samples)
    peak_compiler = max(sample["largest_compiler_rss_gib"] for sample in samples)
    minimum_available = min(sample["available_gib"] for sample in samples)
    assert peak_tree < 16 and peak_compiler < 8 and minimum_available >= 96
    assert manifest["tested_native_actual_exit"] == 0
    assert manifest["cutlass_revision"] == "e87870bb7aacf83bfb3be9da685f148316b8df23"
    assert manifest["normal_package_import_verified"] is False
    assert manifest["accuracy_verified"] is False and manifest["performance_verified"] is False
    return {
        "source_revision": manifest["source_revision"],
        "original_git_archive_and_snapshot_rechecked": True,
        "original_files_checked": checked,
        "only_original_source_delta": ["cmake/BuildFlags.cmake"],
        "manifest_sha256": digest(attempt / "tested-native-source-manifest.json"),
        "compiler_policy_sha256": manifest["compiler_policy_sha256"],
        "actual_exit": 0,
        "all_owned_process_identities_drained": True,
        "native": native,
        "observed_peak_tree_gib": peak_tree,
        "observed_peak_compiler_gib": peak_compiler,
        "observed_minimum_available_gib": minimum_available,
        "normal_package_import_verified": False,
        "accuracy_verified": False,
        "performance_verified": False,
        "official_full_acceptance": False,
    }


def audit_bf(root):
    stage = root / "bf16-gemm/canonical-v4-icpx"
    manifest = load(stage / "source-binary-manifest.json")
    policy = load(stage / "compiler-policy-verified.json")
    frozen = load(root / "bf16-gemm/canonical-source-freeze-v4-icpx.json")
    assert manifest["source"]["source_digest"] == frozen["source_digest"]
    assert load(stage / "source-readonly-postverify.json")["verified"] is True
    assert manifest["build_evidence"]["exit"] == 0 and manifest["install_actual_exit"] == 0
    for phase in ["configure", "default-all-build", "supervisor"]:
        assert (stage / f"{phase}.exit").read_text().strip() == "0"
    assert policy["compiler_id"] == "IntelLLVM" and policy["no_GNU_command"] is True
    for key in ["cxx", "sycl", "host"]:
        assert policy[key] == "/opt/gfx-deps/dpcpp/bin/icpx"
    evidence = manifest["build_evidence"]
    for label in ["command", "exit_file", "log"]:
        file_key = "exit_file" if label == "exit_file" else f"{label}_file"
        assert digest(Path(evidence[file_key])) == evidence[f"{label}_sha256"]
    site = Path(manifest["site"])
    for relative, record in manifest["installed_files"].items():
        assert digest(site / relative) == record["sha256"], relative
    assert manifest["acceptance_complete"] is False
    return {
        "full_default_build_actual_exit": 0,
        "install_actual_exit": 0,
        "source_digest": frozen["source_digest"],
        "source_postverified": True,
        "manifest_sha256": digest(stage / "source-binary-manifest.json"),
        "actual_compiler": "icpx",
        "installed_files_rehashed": len(manifest["installed_files"]),
        "native_binary_inventory": len(manifest["binaries"]),
        "normal_package_runtime_and_performance_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "version": "completed-icpx-cpu-v1",
        "epoch": time.time(),
        "scope": "independent completed CPU evidence audit; no GPU work",
        "source_limited_native": {component: audit_source(args.root, component) for component in ["memory", "position", "logits-vocab"]},
        "bf16_gemm_full_default": audit_bf(args.root),
        "verified": True,
        "final_accuracy_performance_acceptance": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

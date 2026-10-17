import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    audits = json.loads((ROOT / "COMPLETED_ICPX_CPU_AUDIT_v1.json").read_text())
    assert audits["verified"] is True
    for component in ("memory", "position", "logits-vocab"):
        artifact = ROOT / component
        historical_path = artifact / "diagnostic-source-binary-manifest.json"
        historical_manifest = json.loads(historical_path.read_text())
        historical = artifact / "diagnostic/before/site/sgl_kernel"
        expected = historical_manifest["packages"]["before"]
        files = {path.relative_to(historical).as_posix(): digest(path) for path in historical.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
        assert files == expected["package_files"]
        for name, sha256 in expected["registration_extension_sha256"].items():
            assert digest(historical / name) == sha256
        attempt = artifact / "fresh-source-icpx-v1/runs/source-native-icpx-20261017-r1"
        manifest_path = attempt / "tested-native-source-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert digest(manifest_path) == audits["source_limited_native"][component]["manifest_sha256"]
        assert manifest["source_revision"] == historical_manifest["source_revision"]
        assert manifest["tested_native_actual_exit"] == 0
        evidence = artifact / "source-compiler-accuracy-main-v1"
        evidence.mkdir(exist_ok=False)
        roles = {}
        for role in ("historical", "fresh-icpx"):
            destination = evidence / role / "site/sgl_kernel"
            shutil.copytree(historical, destination, ignore=shutil.ignore_patterns("__pycache__"))
            if role == "fresh-icpx":
                for name, record in manifest["native"].items():
                    library = attempt / "build" / name
                    assert digest(library) == record["sha256"]
                    assert (destination / name).is_file()
                    shutil.copy2(library, destination / name)
            for name, sha256 in expected["registration_extension_sha256"].items():
                assert digest(destination / name) == sha256
            roles[role] = {"files": {"sgl_kernel/" + path.relative_to(destination).as_posix(): digest(path) for path in destination.rglob("*") if path.is_file()}, "registration_extension_sha256": expected["registration_extension_sha256"]}
        test_names = {"memory": ("test_paged_alloc.py", "test_write_req_to_token_pool.py"), "position": ("test_compute_position.py",), "logits-vocab": ("test_softcap_logits.py", "test_vocab_mask.py")}[component]
        tests = {}
        with tarfile.open(attempt / "original-source.tar") as archive:
            for name in test_names:
                actual = artifact / "native-diagnostic/tests" / name
                assert actual.read_bytes() == archive.extractfile("tests/" + name).read()
                tests[name] = digest(actual)
        report = {
            "component": component,
            "source_revision": manifest["source_revision"],
            "historical_source_manifest": str(historical_path),
            "historical_source_manifest_sha256": digest(historical_path),
            "fresh_source_manifest": str(manifest_path),
            "fresh_source_manifest_sha256": digest(manifest_path),
            "tested_native": {name: record["sha256"] for name, record in manifest["native"].items()},
            "roles": roles,
            "original_tests_sha256": tests,
            "method": "genuine original normal SOURCE package/registration in both roles; only original tested-native replaced with fresh icpx; existing original and reviewed alias exports retained",
            "original_references_driver_sha256": digest(ROOT / "benchmark_simple.py"),
            "normal_original_registration_preserved": True,
            "no_Torch_provider_copy": True,
            "accuracy_pending": True,
            "timing_authorized": False,
            "AFTER": False,
            "canonical_full_acceptance": False,
        }
        (evidence / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"component": component, "prepared": str(evidence), "manifest_sha256": digest(evidence / "prepared.json"), "original_tests_checked": len(tests)}))


if __name__ == "__main__":
    main()

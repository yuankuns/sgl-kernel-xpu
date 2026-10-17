import hashlib
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    import torch

    assert not torch.xpu.is_initialized()
    assert os.sched_getaffinity(0) == set(range(112, 128))
    grant_path = ROOT / "SIMPLE_SOURCE_COMPILER_ACCURACY_GPU11_GRANT_v2.json"
    grant = load(grant_path)
    assert digest(ROOT / "canonical-tools/simple_source_accuracy_v1.py") == grant["driver_sha256"]
    assert (ROOT / "simple-source-compiler-accuracy-gpu11-v2-supervisor.exit").read_text().strip() == "0"
    assert not Path("/proc/3043288").exists() and not Path("/proc/3043358").exists()
    results = {}
    for component, test_count, case_count in [("memory", 24, 5), ("position", 9, 4), ("logits-vocab", 36, 5)]:
        evidence = ROOT / component / "source-compiler-accuracy-main-v1"
        prepared_path = evidence / "prepared.json"
        prepared = load(prepared_path)
        assert digest(prepared_path) == grant["prepared_sha256"][component]
        assert digest(Path(prepared["fresh_source_manifest"])) == prepared["fresh_source_manifest_sha256"]
        outputs = {}
        roles = {}
        for role in ("historical", "fresh-icpx"):
            stage = evidence / role
            for phase in ("CPU-normal-provider-final", "production-reference", "original-tests"):
                assert (stage / f"{phase}.exit").read_text().strip() == "0"
            tests_log = (stage / "original-tests.log").read_text()
            assert re.search(rf"\b{test_count} passed\b", tests_log)
            assert not re.search(r"\b\d+ (?:skipped|failed|xfailed|xpassed)\b", tests_log)
            report = load(stage / "production-reference.json")
            assert report["normal_original_registration"] is True and report["AFTER"] is False
            assert all(report[field] is False for field in ("timing", "events", "graphs", "warmup"))
            assert len(report["cases"]) == case_count
            for case in report["cases"]:
                assert case["independent_original_full_reference"] == "PASS"
            for relative, sha256 in report["mapped_native"].items():
                assert sha256 == prepared["roles"][role]["files"][relative]
                assert digest(stage / "site" / relative) == sha256
            outputs[role] = torch.load(stage / "checked-full-outputs.pt", map_location="cpu", weights_only=True)
            roles[role] = {"actual_exit": 0, "original_tests_passed": test_count, "production_reference_cases": case_count, "normal_package_reference_report_sha256": digest(stage / "production-reference.json"), "test_log_sha256": digest(stage / "original-tests.log"), "full_output_file_sha256": digest(stage / "checked-full-outputs.pt"), "mapped_native_count": len(report["mapped_native"])}
        assert outputs["historical"].keys() == outputs["fresh-icpx"].keys()
        comparisons = {}
        for case, historical in outputs["historical"].items():
            fresh = outputs["fresh-icpx"][case]
            assert historical.keys() == fresh.keys()
            records = {}
            for name, expected in historical.items():
                actual = fresh[name]
                assert expected.dtype == actual.dtype and expected.shape == actual.shape
                assert torch.equal(expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8))
                records[name] = {"elements": expected.numel(), "dtype": str(expected.dtype), "actual_entire_tensor_bitwise": True}
            comparisons[case] = records
        assert (evidence / "full-output-analysis.exit").read_text().strip() == "0"
        assert load(evidence / "full-output-equivalence.json")["all_full_outputs_bitwise"] is True
        results[component] = {"source_revision": prepared["source_revision"], "roles": roles, "full_output_comparisons": comparisons, "all_original_thresholds_preserved": True, "AFTER": False, "performance_verified": False}
    assert not torch.xpu.is_initialized()
    report = {"version": "simple-source-compiler-accuracy-v1", "epoch": time.time(), "grant_sha256": digest(grant_path), "supervisor_actual_exit": 0, "gpu": 11, "cpu_affinity": list(range(112, 128)), "known_supervisor_and_wrapper_pids_absent": True, "components": results, "CPU_auditor_GPU_initialized": False, "verified": True, "scope": "historical genuine SOURCE original registration and kernels versus original kernels freshly compiled by icpx; NOT AFTER/full port acceptance", "performance_verified": False}
    destination = ROOT / "SIMPLE_SOURCE_COMPILER_ACCURACY_RESULTS_v1.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verified": True, "source_accuracy_components": list(results), "report": str(destination), "AFTER": False, "performance": False}))


if __name__ == "__main__":
    main()

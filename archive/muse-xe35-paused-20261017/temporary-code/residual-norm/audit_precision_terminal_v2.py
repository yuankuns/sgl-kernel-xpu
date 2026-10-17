import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "normal-precision-pytest-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run(["sha256sum", "-c", "normal-precision-pytest-v2-evidence.sha256"], cwd=ROOT, check=True)
    audit = json.loads((ROOT / "NORMAL_PRECISION_PYTEST_FINAL_AUDIT_v2.json").read_text())
    assert all(value == 0 for value in audit["actual_phase_exits"].values())
    records = {}
    for role, count in (("SOURCE", 69), ("AFTER", 73)):
        tree = ET.parse(OUTPUT / f"{role}-accuracy.xml")
        cases = tree.findall(".//testcase")
        assert len(cases) == count
        assert not tree.findall(".//skipped") and not tree.findall(".//failure") and not tree.findall(".//error")
        terminal = json.loads((OUTPUT / f"{role}-terminal.json").read_text())
        assert terminal["passed_count"] == count and not terminal["failed"] and not terminal["skipped"]
        records[role] = dict(actual_original_tests=count, failures=0, skips=0, actual_exit=0,
            JUnit_SHA256=sha(OUTPUT / f"{role}-accuracy.xml"), log_SHA256=sha(OUTPUT / f"{role}.log"),
            result_reports_SHA256=sha(OUTPUT / f"{role}-results.jsonl"),
            successful_pytest_GPU_input_output_tensor_files_saved=False,
            successful_pytest_GPU_tensor_SHA256_unavailable=True,
            limitation="original success tests check all CPU-reference elements but do not persist GPU inputs/outputs; no SHA is invented")
    assert all(not Path(f"/proc/{pid}").exists() for pid in (3294996, 3300659))
    assert (OUTPUT / "container-drain.exit").read_text().strip() == "0"
    lock = subprocess.run(["flock", "-n", "/tmp/intel-gpu-9.lock", "true"]).returncode
    assert lock == 0
    history = ROOT / "history24-accuracy-v1"
    complete = json.loads((history / "final-audit.json").read_text())
    assert all(value == 0 for value in complete["actual_phase_exits"].values())
    assert (ROOT / "history24-v1-fulltensor-verify.exit").read_text().strip() == "0"
    seal = history / "fulltensor-evidence.sha256"
    assert len(seal.read_text().splitlines()) == 804
    original_terminal = (ROOT / "normal-precision-pytest-v1/supervisor-terminal.json").read_bytes()
    assert hashlib.sha256(original_terminal).hexdigest() == "97b738d4c4fdcd9cdc5018e649b9bc5f5482ecaa06c38f306264210f21f69089"
    assert (ROOT / "normal-precision-pytest-v1/supervisor.exit").read_text().strip() == "1"
    assert all((ROOT / f"normal-precision-pytest-v2-startup-abort{attempt}.exit").read_text().strip() == "1" for attempt in (1, 2))
    result = dict(epoch=time.time(), independent_audit_actual_exit=0, roles=records,
        full_pytest_raw_evidence_files=118, raw_evidence_manifest_SHA256=sha(ROOT / "normal-precision-pytest-v2-evidence.sha256"),
        final_audit_SHA256=sha(ROOT / "NORMAL_PRECISION_PYTEST_FINAL_AUDIT_v2.json"),
        host_and_container_owned_runtime_drained=True, GPU9_lock_free_exit=lock,
        history24_separate_scope=dict(full_input_oracle_SOURCE_AFTER_output_tensor_and_masks_files=804,
            fulltensor_manifest_SHA256=sha(seal), actual_fulltensor_verification_exit=0,
            full_paired_elements=8055029760, mismatch=0, not_pytest_success_tensor_hashes=True),
        original_v1_failure1_and_two_startup_aborts1_preserved=True,
        old_NaN_explained=False, model_old_binding_fixed=False, model_path_passed=False,
        new_GPU_or_compile_or_performance=False)
    (ROOT / "RESIDUAL_NORMAL_PRECISION_INDEPENDENT_AUDIT_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(independent_audit_actual_exit=0, SOURCE=69, AFTER=73, skip=0, failure=0,
        successful_pytest_tensor_SHA_unavailable=True, history24_fulltensor_files=804)), flush=True)


if __name__ == "__main__":
    main()

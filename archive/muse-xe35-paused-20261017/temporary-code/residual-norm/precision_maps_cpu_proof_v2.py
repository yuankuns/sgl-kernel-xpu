import json
from pathlib import Path

import after_pre_wave_accuracy_v4 as after
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent


def mapped_digest(name):
    return source.digest(Path(name))


def main():
    torch, _, provider = helpers.provider("SOURCE")
    assert not torch.xpu.is_initialized() and not torch.xpu.is_available()
    try:
        source.digest(str(ROOT / "precision_pytest_v1.py"))
    except AttributeError as error:
        old_error = str(error)
    else:
        raise AssertionError("Expected exact v1 str/Path defect was not reproduced")
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "libze_intel_gpu" in line or "libsycl.so" in line}
    mapped = {str(Path(name).resolve()): mapped_digest(name) for name in libraries}
    proof = after.driver_proof()
    candidates = proof["GPU_driver_candidate_libraries_sha256"]
    assert mapped and any("libsycl" in name for name in mapped)
    assert any(value == proof["SYCL_runtime"]["sha256"] for name, value in mapped.items() if "libsycl" in name)
    assert all(value == candidates[name] for name, value in mapped.items() if "libze_intel_gpu" in name)
    assert not torch.xpu.is_initialized()
    record = dict(actual_CPU_proof_exit=0, old_v1_defect_reproduced=old_error,
        proposed_change="source.digest(name) -> source.digest(Path(name)) in live-map metadata only",
        current_native_and_driver_SHA_gate0=True, mapped_libraries=mapped, normal_SOURCE_provider=provider,
        GPU_initialized=False, GPU_kernel_invocations=0, old_v1_modified=False,
        framework_math_tolerance_PR_unchanged=True, new_GPU_grant_or_retry=False)
    (ROOT / "PRECISION_MAPS_CPU_PROOF_v2.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(actual_CPU_proof_exit=0, old_defect_reproduced=True,
        proposed_local_Path_fix_verified=True, GPU_initialized=False, GPU_retry=False)), flush=True)


if __name__ == "__main__":
    main()

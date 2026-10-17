import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import history24_stage as monitor
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "normal-precision-pytest-v2"
ORIGINAL = ROOT / "normal-precision-pytest-v1"


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def mechanical():
    records = {}
    names = ("precision_pytest_v1.py", "precision_pytest_stage_v1.py", "precision_pytest_container_v1.sh", "run_precision_pytest_v1.py")
    replacements = {name: name.replace("_v1", "_v2") for name in names}
    replacements["normal-precision-pytest-v1"] = "normal-precision-pytest-v2"
    for original, current in replacements.items():
        if not original.endswith((".py", ".sh")):
            continue
        previous = (ROOT / original).read_text()
        expected = previous
        for old, new in replacements.items():
            expected = expected.replace(old, new)
        expected = expected.replace("import precision_pytest_v1 as check", "import precision_pytest_v2 as check")
        if original == "precision_pytest_v1.py":
            expected = expected.replace("source.digest(name) for name in driver", "source.digest(Path(name)) for name in driver")
            expected = expected.replace('"run_precision_pytest_v2.py")', '"run_precision_pytest_v2.py", "precision_maps_preflight_v2.py")')
            expected = expected.replace("    return record\n\n\ndef grant():", '    import precision_maps_preflight_v2 as metadata\n    record["CPU_metadata_preflight"] = metadata.binding()\n    return record\n\n\ndef grant():')
        elif original == "precision_pytest_container_v1.sh":
            expected = expected.replace('  stage CPU-binding "$python" binding', '  "$python" -B "$artifact/precision_maps_preflight_v2.py"\n  stage CPU-binding "$python" binding')
        elif original == "run_precision_pytest_v1.py":
            expected = expected.replace('    binding = json.loads((check.OUTPUT / "binding.json").read_text())', '    assert all(not os.path.exists(f"/proc/{pid}") for pid in (3278336, 3283576))\n    binding = json.loads((check.OUTPUT / "binding.json").read_text())')
        actual = (ROOT / current).read_text()
        assert actual == expected
        previous_seal = json.loads((ROOT / "RESIDUAL_PRECISION_PYTEST_WRITE_SET_v1.json").read_text())
        sealed = next(record for record in previous_seal["authored_and_audit_files"] if record["path"] == original)
        assert source.digest(ROOT / original) == sealed["sha256"]
        if original.endswith(".py"):
            ast.parse(actual)
        diff = "".join(difflib.unified_diff(previous.splitlines(keepends=True), actual.splitlines(keepends=True), original, current))
        (OUTPUT / (current + ".mechanical.diff")).write_text(diff)
        records[current] = dict(original_SHA256=source.digest(ROOT / original), current_SHA256=source.digest(ROOT / current),
            diff_SHA256=hashlib.sha256(diff.encode()).hexdigest(), only_approved_runtime_Path_fix_and_version_or_gate_binding=True)
    return records


def binding():
    assert (OUTPUT / "CPU-metadata-preflight.exit").read_text().strip() == "0"
    proof = json.loads((OUTPUT / "CPU-metadata-preflight.json").read_text())
    result = dict(metadata_preflight_sha256=source.digest(OUTPUT / "CPU-metadata-preflight.json"),
        mechanicaldiff=proof["mechanicaldiff"], old_owned_host_and_container_PIDs_absent=proof["old_owned_PIDs_absent"])
    for role in ("SOURCE", "AFTER"):
        assert (OUTPUT / f"CPU-{role}-maps.exit").read_text().strip() == "0"
        role_proof = json.loads((OUTPUT / f"CPU-{role}-maps.json").read_text())
        assert not role_proof["GPU_initialized"] and role_proof["Path_actual_metadata_preflight0"]
        new = json.loads((OUTPUT / f"CPU-{role}.json").read_text())
        old = json.loads((ORIGINAL / f"CPU-{role}.json").read_text())
        assert new["nodeids"] == old["nodeids"] == role_proof["original_nodeids"]
        assert new["provider_maps"] == role_proof["provider_maps"]
        assert new["driver_proof"] == role_proof["driver_proof"]
        assert new["test_sha256"] == old["test_sha256"] == role_proof["test_sha256"]
        result[role] = dict(metadata_SHA256=source.digest(OUTPUT / f"CPU-{role}-maps.json"),
            original_nodeids_SHA256=hashlib.sha256(json.dumps(old["nodeids"]).encode()).hexdigest(),
            original_test_SHA256=old["test_sha256"], count=new["count"])
    return result


def child(role):
    assert os.environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:cpu"
    assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == "/opt/gfx-deps/profiling_tools/lib"
    import after_pre_wave_accuracy_v4 as after
    torch, _, maps = helpers.provider(role)
    assert not torch.xpu.is_initialized() and not torch.xpu.is_available()
    collected = json.loads((OUTPUT / f"CPU-{role}.json").read_text())
    original = json.loads((ORIGINAL / f"CPU-{role}.json").read_text())
    assert collected["nodeids"] == original["nodeids"] and collected["count"] == (69 if role == "SOURCE" else 73)
    assert maps == collected["provider_maps"]
    driver = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if "libze_intel_gpu" in line or "libsycl.so" in line}
    mapped = {str(Path(name).resolve()): source.digest(Path(name)) for name in driver}
    driver_proof = after.driver_proof()
    candidates = driver_proof["GPU_driver_candidate_libraries_sha256"]
    assert any("libze_intel_gpu" in name for name in mapped)
    assert all(value == candidates[name] for name, value in mapped.items() if "libze_intel_gpu" in name)
    assert any(value == driver_proof["SYCL_runtime"]["sha256"] for name, value in mapped.items() if "libsycl" in name)
    assert driver_proof == collected["driver_proof"]
    probe = torch.load(OUTPUT / f"{role}-CPU-metadata-probe.pt", weights_only=True)
    recorded = json.loads((OUTPUT / f"{role}-CPU-metadata-probe.json").read_text())
    metadata = helpers.metadata(torch, probe)
    assert source.digest(OUTPUT / f"{role}-CPU-metadata-probe.pt") == recorded["file_sha256"]
    for name, value in metadata.items():
        for field in ("dtype", "shape", "stride", "storage_offset", "sha256"):
            assert value[field] == recorded["metadata"][name][field]
    assert not torch.xpu.is_initialized()
    write(f"CPU-{role}-maps.json", dict(provider_maps=maps, libraries_SHA256=mapped, driver_proof=driver_proof,
        original_nodeids=original["nodeids"], test_sha256=collected["test_sha256"], whole_metadata_file_and_tensor_digest0=True,
        Path_actual_metadata_preflight0=True, GPU_initialized=False, actual_kernel_invocations=0))


def supervise(role):
    interpreter = ROOT / ("source-before-icpx-bounded-v1/isolated-source-venv/bin/python" if role == "SOURCE" else "source-full-venv/bin/python")
    environment = dict(os.environ)
    if role == "SOURCE":
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(ROOT / "canonical-v4-icpx/site")
    argv = [str(interpreter), "-B", str(Path(__file__).resolve()), "--role", role]
    phase = f"CPU-{role}-maps"
    write(phase + ".command.json", argv)
    peak_worker = peak_tree = 0
    violation = False
    with (OUTPUT / (phase + ".log")).open("w") as logfile:
        process = subprocess.Popen(argv, env=environment, stdout=logfile, stderr=subprocess.STDOUT)
        (OUTPUT / (phase + ".pid")).write_text(str(process.pid) + "\n")
        with (OUTPUT / (phase + "-RSS.jsonl")).open("w") as samples:
            while process.poll() is None:
                current = monitor.processes(process.pid)
                worker = current.get(process.pid, {}).get("rss", 0)
                tree = sum(record["rss"] for record in current.values())
                peak_worker, peak_tree = max(peak_worker, worker), max(peak_tree, tree)
                samples.write(json.dumps(dict(epoch=time.time(), processes=current, worker_RSS=worker, tree_RSS=tree)) + "\n")
                samples.flush()
                if worker > 64 * 2**30 or tree > 80 * 2**30:
                    violation = True
                    monitor.terminate(process, current)
                    break
                time.sleep(0.5)
        status = process.wait()
    (OUTPUT / (phase + ".exit")).write_text(str(status) + "\n")
    write(phase + "-supervisor.json", dict(actual_exit=status, violation=violation,
        peak_worker_RSS=peak_worker, peak_tree_RSS=peak_tree, worker_cap_GiB=64, tree_cap_GiB=80))
    if status or violation:
        raise RuntimeError(phase + " failed; no next provider or grant")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", choices=("SOURCE", "AFTER"))
    args = parser.parse_args()
    source.scope()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if args.role:
        child(args.role)
        return
    status = 1
    try:
        old_pids = (3278336, 3283576, 268322, 268321)
        assert all(not Path(f"/proc/{pid}").exists() for pid in old_pids)
        mechanicaldiff = mechanical()
        for role in ("SOURCE", "AFTER"):
            assert (OUTPUT / f"CPU-{role}.exit").read_text().strip() == "0"
            supervise(role)
        write("CPU-metadata-preflight.json", dict(mechanicaldiff=mechanicaldiff,
            old_owned_PIDs_absent=old_pids, original69_73_nodeids_unmodified=True,
            no_kernel_math_tolerance_filter_change=True, GPU_initialized=False))
        status = 0
    finally:
        (OUTPUT / "CPU-metadata-preflight.exit").write_text(str(status) + "\n")


if __name__ == "__main__":
    main()

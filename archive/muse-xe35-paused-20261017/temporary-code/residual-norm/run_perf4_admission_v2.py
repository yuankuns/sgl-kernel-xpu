import json
import os
from pathlib import Path
import subprocess
import sys
import time

import perf4_admission_v2 as check


WRAPPER = "/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh"
CONTAINER = "/root/" + str(check.ROOT).removeprefix("/home/syk/")


def command(name, argv):
    check.write(name + ".command.json", argv)
    with (check.OUTPUT / (name + ".log")).open("w") as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / (name + ".pid")).write_text(str(process.pid) + "\n")
        status = process.wait()
    (check.OUTPUT / (name + ".exit")).write_text(str(status) + "\n")
    assert status == 0, name + " failed; no next phase, no retry"


def container_command(mode, extra=None, role=None):
    python = CONTAINER + ("/source-before-icpx-bounded-v1/isolated-source-venv/bin/python" if role == "SOURCE" else "/source-full-venv/bin/python")
    selector = "level_zero:cpu"
    code = 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}" PYTHONDONTWRITEBYTECODE=1 ONEAPI_DEVICE_SELECTOR=' + selector + '; unset PYTHONPATH; '
    if role == "AFTER":
        code += 'export PYTHONPATH="' + CONTAINER + '/canonical-v4-icpx/site"; '
    argv = [WRAPPER, "--workdir", CONTAINER, "taskset", "-c", "48-63", "bash", "-c", code + 'exec "$@"', "bash", python, "-B", CONTAINER + "/perf4_admission_v2.py", "--mode", mode]
    return argv + (extra or [])


def budget(label):
    stat = os.statvfs(check.perf.OUTPUT)
    available = stat.f_bavail * stat.f_frsize
    other_reserved = (10 + 32) * 2**30
    assert available - other_reserved >= 740 * 2**30
    record = dict(epoch=time.time(), available_bytes=available, other_large_parallel_reservations_bytes=other_reserved,
        other_reservation_basis="main release: QK6 and FM5 terminal/drained; their consumed raw already reflected in df, unconsumed QK380 reservation released; conservatively reserve NV11 candidate10GiB plus future ungranted FM32GiB; existing allocated raw already reflected in df",
        available_after_other_reservations_bytes=available - other_reserved,
        success_estimate_bytes=576 * 2**30, residual_reservation_bytes=640 * 2**30,
        required_success_plus_headroom_bytes=676 * 2**30, headroom_bytes=100 * 2**30,
        conservative_cap_plus_headroom_bytes=740 * 2**30,
        remaining_after_residual_cap_bytes=available - other_reserved - 640 * 2**30)
    check.write(label + "-disk-reservation.json", record)
    return record


def quiet(label):
    expected = check.stable_stack()
    start = time.monotonic()
    with (check.OUTPUT / (label + "-host-quiet.jsonl")).open("x") as logfile:
        process = subprocess.Popen(container_command("quiet", ["--label", label]),
            stdout=(check.OUTPUT / (label + "-container-quiet.log")).open("w"), stderr=subprocess.STDOUT)
        while process.poll() is None:
            active = check.active_maintenance()
            logfile.write(json.dumps(dict(epoch=time.time(), active=active, stack=check.stable_stack())) + "\n")
            logfile.flush()
            if active or check.stable_stack() != expected:
                process.terminate()
                process.wait()
                raise RuntimeError("host maintenance/stack changed; no retry")
            time.sleep(0.5)
        status = process.wait()
    (check.OUTPUT / (label + "-quiet.exit")).write_text(str(status) + "\n")
    assert status == 0 and time.monotonic() - start >= 60
    check.write(label + "-quiet-complete.json", dict(actual_exit=0, seconds=time.monotonic() - start,
        host_and_container_non_zombie_compiler_pkgmanager_scans=True))


def main():
    check.source.scope()
    assert len(sys.argv) == 1 and not check.perf.BARRIER.exists()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    check.OUTPUT.mkdir()
    (check.OUTPUT / "host-supervisor.pid").write_text(str(os.getpid()) + "\n")
    os.environ["SGLANG_MUSE_CONTAINER"] = "codex-muse-bf16-review-20261015"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for role in ("SOURCE", "AFTER"):
        command("fresh-" + role, container_command("provider", ["--role", role], role))
    quiet("pregrant")
    subprocess.run(["flock", "-n", "/tmp/intel-gpu-4.lock", "true"], check=True)
    command("physical4-process-drain", container_command("snapshot", ["--label", "pregrant-drain"]))
    disk = budget("pregrant")
    ready = json.loads((check.ROOT / "GPU4_NORMAL_FULL_PERF_READY_v1.json").read_text())
    dataset = json.loads((check.perf.OUTPUT / "CPU-dataset.json").read_text())
    for role in ("SOURCE", "AFTER"):
        fresh = json.loads((check.OUTPUT / ("fresh-" + role + ".json")).read_text())
        assert fresh["binding"] == ready["bindings"] == dataset["binding"]
    issued = time.time()
    grant = dict(issuer="main coordinator", coordinator_release=True, phase=check.perf.PHASE,
        gpu=4, CPU="48-63", issued_epoch=issued, start_epoch=issued + 120,
        performance_granted=True, native_compilers_drained=True, stable_window_granted=True,
        cases=24, BAAB_rounds=3, graph_calls=[128, 256], samples=31,
        acceptance_scope="original operator contract", binding=ready["bindings"],
        CPU_provider_SHA256=ready["CPU_provider_SHA256"], CPU_dataset_SHA256=ready["dataset_SHA256"],
        admission_supervisor_SHA256=check.source.digest(check.ROOT / "perf4_admission_v2.py"),
        host_admission_SHA256=check.source.digest(Path(__file__)),
        fresh_provider_SHA256={role:check.source.digest(check.OUTPUT / ("fresh-" + role + ".json")) for role in ("SOURCE", "AFTER")},
        quiet60_SHA256=check.source.digest(check.OUTPUT / "pregrant-quiet-complete.json"),
        disk_reservation=disk, worker_RSS_GiB=96, tree_RSS_GiB=128, artifact_cap_GiB=640,
        disk_headroom_GiB=100, timeout_hours=12, no_retry=True, case18_included=True,
        old_missing_failedraw_replayed=False, old_NaN_explained=False, model_bridge_passed=False)
    patch = "*** Begin Patch\n*** Add File: " + str(check.perf.BARRIER) + "\n" + "".join("+" + line + "\n" for line in json.dumps(grant, indent=2).splitlines()) + "*** End Patch\n"
    subprocess.run(["apply_patch", patch], check=True)
    check.perf.BARRIER.chmod(0o444)
    grant_sha = check.source.digest(check.perf.BARRIER)
    check.write("launch-admission.json", dict(grant_SHA256=grant_sha, grant=grant, host_pid=os.getpid(), stage="future+120 waiting"))
    print(json.dumps(dict(host_PID=os.getpid(), issued=issued, epoch=grant["start_epoch"], grant_SHA256=grant_sha, disk=disk)), flush=True)
    while time.time() < grant["start_epoch"] - 65:
        assert not check.active_maintenance()
        time.sleep(0.5)
    quiet("prelaunch")
    while time.time() < grant["start_epoch"]:
        assert not check.active_maintenance()
        time.sleep(0.5)
    assert check.source.digest(check.perf.BARRIER) == grant_sha
    budget("prelaunch")
    subprocess.run(["flock", "-n", "/tmp/intel-gpu-4.lock", "true"], check=True)
    command("physical4-prelaunch-drain", container_command("snapshot", ["--label", "prelaunch-drain"]))
    python = CONTAINER + "/source-full-venv/bin/python"
    code = 'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; unset PYTHONPATH; exec "$@"'
    command("GPU4-wrapper", [WRAPPER, "--gpu-lock", "--gpu", "4", "--workdir", CONTAINER,
        "taskset", "-c", "48-63", "bash", "-c", code, "bash", python, "-B", CONTAINER + "/perf4_admission_v2.py", "--mode", "runtime"])


if __name__ == "__main__":
    status = 1
    try:
        main()
        status = 0
    finally:
        if check.OUTPUT.exists():
            (check.OUTPUT / "supervisor.exit").write_text(str(status) + "\n")
    sys.exit(status)

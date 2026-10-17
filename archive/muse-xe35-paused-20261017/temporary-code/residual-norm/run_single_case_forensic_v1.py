import json
import os
from pathlib import Path
import subprocess
import sys
import time

import single_case_forensic_v1 as check


OWN_OUTPUT = False


def command(name, argv):
    check.write(f"{name}.command.json", argv)
    with (check.OUTPUT / f"{name}.log").open("w") as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / f"{name}.pid").write_text(str(process.pid) + "\n")
        status = process.wait()
    (check.OUTPUT / f"{name}.exit").write_text(str(status) + "\n")
    return status


def main():
    global OWN_OUTPUT
    check.source.scope()
    if len(sys.argv) != 1:
        raise ValueError("One-case accuracy only; no batch/timing overrides")
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    assert not check.BARRIER.exists()
    check.OUTPUT.mkdir()
    OWN_OUTPUT = True
    (check.OUTPUT / "host-supervisor.pid").write_text(str(os.getpid()) + "\n")
    os.environ["SGLANG_MUSE_CONTAINER"] = "codex-muse-bf16-review-20261015"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    container = "/root/" + str(check.ROOT).removeprefix("/home/syk/")
    wrapper = "/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh"
    script = container + "/single_case_forensic_v1_container.sh"
    preflight = command("CPU-wrapper", [wrapper, "--workdir", container, "taskset", "-c", "48-63", "bash", script, "preflight"])
    if preflight:
        raise RuntimeError("CPU gate failed; no grant/GPU")
    record = json.loads((check.OUTPUT / "binding.json").read_text())
    assert all((check.OUTPUT / f"{name}.exit").read_text().strip() == "0"
               for name in ("CPU-Torch", "CPU-SOURCE", "CPU-AFTER", "CPU-binding", "CPU-wrapper"))
    check.write("ready.json", dict(record, host_pid=os.getpid(), ready_epoch=time.time(), GPU_started=False))
    subprocess.run(["flock", "-n", "/tmp/intel-gpu-7.lock", "true"], check=True)
    released = dict(record, issuer="main coordinator user authorization", gpu=7, CPU="48-63",
                    phase=check.PHASE, coordinator_release=True, accuracy_only=True,
                    scope="B8/ISL4096/OSL16/TP1/EP1", case="32768_weight_0_1",
                    start_epoch=time.time() + 120, no_timing_events_graphs_warmup=True,
                    invocations_per_provider=1, formal_perf_barrier_consumed=False,
                    original_failure_actual_exit=1, new_inputs_not_missing_failed_raw=True,
                    tolerances=dict(norm=5e-3, residual=1e-5, widen=1e-5, widen_exact=True))
    patch = "*** Begin Patch\n*** Add File: " + str(check.BARRIER) + "\n"
    patch += "".join("+" + line + "\n" for line in json.dumps(released, indent=2).splitlines())
    subprocess.run(["apply_patch", patch + "*** End Patch\n"], check=True)
    check.BARRIER.chmod(0o444)
    print(json.dumps(dict(handle="single-case-forensic-v1", host_pid=os.getpid(), grant=str(check.BARRIER),
                         epoch=released["start_epoch"], stage="CPU0; fresh+120 grant waiting")), flush=True)
    while time.time() < released["start_epoch"]:
        time.sleep(1)
    assert json.loads(check.BARRIER.read_text()) == released
    assert all(check.source.digest(check.ROOT / name) == value for name, value in record["script_sha256"].items())
    check.write("consumed-grant.json", released)
    status = command("GPU7-wrapper", [wrapper, "--gpu-lock", "--gpu", "7", "--workdir", container,
                                     "taskset", "-c", "48-63", "bash", script, "runtime"])
    with (check.OUTPUT / "GPU7-wrapper.log").open() as logfile:
        print(logfile.read()[-5000:], flush=True)
    return status


if __name__ == "__main__":
    status = 1
    try:
        status = main()
    finally:
        if OWN_OUTPUT:
            (check.OUTPUT / "supervisor.exit").write_text(str(status) + "\n")
            check.write("supervisor-terminal.json", dict(epoch=time.time(), actual_exit=status,
                        host_pid=os.getpid(), accuracy_only=True, performance_approved=False))
    sys.exit(status)

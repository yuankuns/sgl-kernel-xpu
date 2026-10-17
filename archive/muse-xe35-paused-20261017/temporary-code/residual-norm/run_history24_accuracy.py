import json
import os
from pathlib import Path
import subprocess
import sys
import time

import history24_accuracy as check


OWN_OUTPUT = False


def command(name, argv):
    check.write(name + ".command.json", argv)
    with (check.OUTPUT / (name + ".log")).open("w") as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / (name + ".pid")).write_text(str(process.pid) + "\n")
        status = process.wait()
    (check.OUTPUT / (name + ".exit")).write_text(str(status) + "\n")
    return status


def main():
    global OWN_OUTPUT
    check.source.scope()
    if len(sys.argv) != 1:
        raise ValueError("B8-only accuracy24; accepts no timing or batch overrides")
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64)) and not check.BARRIER.exists()
    check.OUTPUT.mkdir()
    OWN_OUTPUT = True
    (check.OUTPUT / "host-supervisor.pid").write_text(str(os.getpid()) + "\n")
    os.environ["SGLANG_MUSE_CONTAINER"] = "codex-muse-bf16-review-20261015"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    container = "/root/" + str(check.ROOT).removeprefix("/home/syk/")
    wrapper = "/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh"
    script = container + "/history24_container.sh"
    if command("CPU-wrapper", [wrapper, "--workdir", container, "taskset", "-c", "48-63", "bash", script, "preflight"]):
        raise RuntimeError("CPU gate failed; no grant/GPU")
    assert all((check.OUTPUT / (name + ".exit")).read_text().strip() == "0" for name in ("CPU-smoke", "CPU-SOURCE", "CPU-AFTER", "CPU-binding", "CPU-wrapper"))
    subprocess.run(["flock", "-n", "/tmp/intel-gpu-7.lock", "true"], check=True)
    record = json.loads((check.OUTPUT / "binding.json").read_text())
    ready = dict(record, host_pid=os.getpid(), ready_epoch=time.time(), GPU_started=False, physical7_lock_free=True)
    check.write("ready.json", ready)
    release = dict(record, issuer="main coordinator user authorization", phase=check.PHASE, gpu=7, CPU="48-63",
                   scope="B8/ISL4096/OSL16/TP1/EP1", coordinator_release=True, accuracy_only=True,
                   cases=24, generator_once=True, original_op_invocations_per_role=24,
                   start_epoch=time.time() + 120, no_timing_events_graphs_warmup=True,
                   formal_perf_barrier_consumed=False, precision_pytest_granted=False,
                   original_failure_actual_exit=1, exact_raw_allocator_replay=False,
                   tolerances=dict(norm=5e-3, stream=1e-5, widen=1e-5, widen_exact=True),
                   disk_cap_GiB=256, disk_headroom_GiB=100, worker_RSS_GiB=16, descendant_tree_RSS_GiB=24)
    patch = "*** Begin Patch\n*** Add File: " + str(check.BARRIER) + "\n"
    patch += "".join("+" + line + "\n" for line in json.dumps(release, indent=2).splitlines())
    subprocess.run(["apply_patch", patch + "*** End Patch\n"], check=True)
    check.BARRIER.chmod(0o444)
    print(json.dumps(dict(handle="history24-accuracy-v1", host_pid=os.getpid(), epoch=release["start_epoch"], grant=str(check.BARRIER), stage="CPU0; future+120 waiting")), flush=True)
    while time.time() < release["start_epoch"]:
        time.sleep(1)
    assert json.loads(check.BARRIER.read_text()) == release
    assert all(check.source.digest(check.ROOT / name) == value for name, value in release["script_sha256"].items())
    check.write("consumed-grant.json", release)
    status = command("GPU7-wrapper", [wrapper, "--gpu-lock", "--gpu", "7", "--workdir", container, "taskset", "-c", "48-63", "bash", script, "runtime"])
    print((check.OUTPUT / "GPU7-wrapper.log").read_text()[-6000:], flush=True)
    return status


if __name__ == "__main__":
    status = 1
    try:
        status = main()
    finally:
        if OWN_OUTPUT:
            (check.OUTPUT / "supervisor.exit").write_text(str(status) + "\n")
            check.write("supervisor-terminal.json", dict(actual_exit=status, host_pid=os.getpid(), epoch=time.time(), accuracy_only=True))
    sys.exit(status)

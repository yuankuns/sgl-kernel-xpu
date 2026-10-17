import json
import os
import subprocess
import sys
import time

import precision_pytest_v2 as check


def command(name, argv):
    check.write(name + ".command.json", argv)
    with (check.OUTPUT / (name + ".log")).open("w") as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / (name + ".pid")).write_text(str(process.pid) + "\n")
        status = process.wait()
    (check.OUTPUT / (name + ".exit")).write_text(str(status) + "\n")
    return status


def main():
    check.source.scope()
    if len(sys.argv) != 1:
        raise ValueError("B8-only precision accuracy; no batch/timing overrides")
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64)) and not check.BARRIER.exists()
    check.OUTPUT.mkdir()
    (check.OUTPUT / "host-supervisor.pid").write_text(str(os.getpid()) + "\n")
    os.environ["SGLANG_MUSE_CONTAINER"] = "codex-muse-bf16-review-20261015"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    container = "/root/" + str(check.ROOT).removeprefix("/home/syk/")
    wrapper = "/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh"
    script = container + "/precision_pytest_container_v2.sh"
    status = command("CPU-wrapper", [wrapper, "--workdir", container, "taskset", "-c", "48-63", "bash", script, "preflight"])
    if status:
        return status
    assert all((check.OUTPUT / (name + ".exit")).read_text().strip() == "0" for name in ("CPU-SOURCE", "CPU-AFTER", "CPU-binding", "CPU-wrapper"))
    subprocess.run(["flock", "-n", "/tmp/intel-gpu-9.lock", "true"], check=True)
    drain = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    check.write("physical9-drain-snapshot.json", dict(epoch=time.time(), lock_probe_exit=0,
        process_snapshot=drain, coordinator_reports_NV9_physically_drained=True))
    assert all(not os.path.exists(f"/proc/{pid}") for pid in (3278336, 3283576))
    binding = json.loads((check.OUTPUT / "binding.json").read_text())
    release = dict(binding, issuer="main coordinator user authorization", phase=check.PHASE,
        coordinator_release=True, gpu=9, CPU="48-63", accuracy_only=True,
        scope="B8/ISL4096/OSL16/TP1/EP1", start_epoch=time.time() + 120,
        no_timing_events_graphs_warmup=True, performance_granted=False, original_tests_once=True,
        stop_on_first_failure=True, worker_RSS_GiB=64, descendant_tree_RSS_GiB=80,
        original_24_failure_actual_exit=1, old_NaN_explained=False)
    check.write("ready.json", release)
    patch = "*** Begin Patch\n*** Add File: " + str(check.BARRIER) + "\n"
    patch += "".join("+" + line + "\n" for line in json.dumps(release, indent=2).splitlines())
    subprocess.run(["apply_patch", patch + "*** End Patch\n"], check=True)
    check.BARRIER.chmod(0o444)
    print(json.dumps(dict(handle="normal-precision-pytest-v2", host_pid=os.getpid(),
        epoch=release["start_epoch"], grant=str(check.BARRIER), stage="CPU gates0; future+120")), flush=True)
    while time.time() < release["start_epoch"]:
        time.sleep(1)
    assert json.loads(check.BARRIER.read_text()) == release
    for name, digest in release["script_sha256"].items():
        assert check.source.digest(check.ROOT / name) == digest
    check.write("consumed-grant.json", release)
    return command("GPU9-wrapper", [wrapper, "--gpu-lock", "--gpu", "9", "--workdir", container,
        "taskset", "-c", "48-63", "bash", script, "runtime"])


if __name__ == "__main__":
    status = 1
    try:
        status = main()
    finally:
        if check.OUTPUT.exists():
            (check.OUTPUT / "supervisor.exit").write_text(str(status) + "\n")
            check.write("supervisor-terminal.json", dict(actual_exit=status, host_pid=os.getpid(),
                epoch=time.time(), accuracy_only=True, no_performance=True))
    sys.exit(status)

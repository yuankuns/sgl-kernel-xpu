import argparse
import idle_waiter_classifier_v3 as waiter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import history24_accuracy as history
import history24_stage as monitor
import residual_normal_full_perf as perf
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "normal-full-perf-gpu4-admission-v3"
NAMES = {"apt", "apt-get", "dpkg", "dkms", "make", "ninja", "cmake", "gcc", "g++", "cc1", "cc1plus",
         "icpx", "icx", "clang", "clang++", "sycl-post-link", "llvm-link", "llvm-spirv", "ld", "ld.lld", "pip", "pip3"}


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def active_maintenance():
    active = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            name = (directory / "comm").read_text().strip()
            argv = (directory / "cmdline").read_bytes().split(b"\0")
            args = [value.decode(errors="replace") for value in argv if value]
            executable = Path(args[0]).name if args else name
            compiler = name in NAMES or executable in NAMES or name.startswith(("clang-", "unattended-upgr"))
            package = executable in ("uv", "conda", "mamba", "micromamba") and any(value in args for value in ("install", "sync", "update", "remove"))
            package |= executable.startswith("python") and any(value in args for value in ("pip", "pip3"))
            if compiler or package:
                if args == waiter.ARGV and waiter.observed_idle(int(directory.name), OUTPUT):
                    continue
                active[int(directory.name)] = dict(name=name, args=args, state=stat[0], starttick=stat[19], ppid=int(stat[1]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return active


def stable_stack():
    stack = history.stack()
    stack.pop("loaded_xe")
    return stack


def active_gpu4():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            environment = dict(value.split(b"=", 1) for value in (directory / "environ").read_bytes().split(b"\0") if b"=" in value)
            if environment.get(b"ZE_AFFINITY_MASK") == b"4" and environment.get(b"ONEAPI_DEVICE_SELECTOR") != b"level_zero:cpu":
                result[int(directory.name)] = dict(starttick=stat[19], args=(directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    return result


def provider(role):
    assert os.environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:cpu"
    torch, _, maps = helpers.provider(role)
    assert not torch.xpu.is_initialized()
    original = json.loads((perf.OUTPUT / f"CPU-{role}.json").read_text())
    assert maps == original["maps"] and perf.prerequisites() == original["binding"]
    write(f"fresh-{role}.json", dict(maps=maps, binding=original["binding"], GPU_initialized=False,
        frozen_provider_SHA256=source.digest(perf.OUTPUT / f"CPU-{role}.json"), actual_provider_gate=0))


def quiet(label):
    expected = stable_stack()
    start = time.monotonic()
    with (OUTPUT / (label + "-quiet.jsonl")).open("x") as logfile:
        while True:
            active = active_maintenance()
            current = stable_stack()
            logfile.write(json.dumps(dict(epoch=time.time(), active=active, loaded_stack=current)) + "\n")
            logfile.flush()
            assert not active and current == expected, "maintenance or stack change; stop, no retry"
            if time.monotonic() - start >= 60:
                break
            time.sleep(0.5)
    write(label + "-quiet.json", dict(actual_exit=0, quiet_seconds=time.monotonic() - start,
        namespace="container", scan_period_seconds=0.5, loaded_stack=expected))


def supervise():
    released = json.loads(perf.BARRIER.read_text())
    assert time.time() >= released["start_epoch"]
    assert source.digest(perf.BARRIER) == json.loads((OUTPUT / "launch-admission.json").read_text())["grant_SHA256"]
    assert not active_maintenance()
    command = ["bash", str(ROOT / "residual_normal_full_perf_container.sh"), "runtime"]
    write("runtime.command.json", command)
    clock = time.monotonic()
    violation = None
    peak_worker = peak_tree = 0
    with (OUTPUT / "runtime.log").open("w") as logfile:
        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
        (OUTPUT / "runtime.pid").write_text(str(process.pid) + "\n")
        with (OUTPUT / "runtime-resource.jsonl").open("w") as samples:
            while process.poll() is None:
                current = monitor.processes(process.pid)
                tree = sum(record["rss"] for record in current.values())
                worker = max((record["rss"] for record in current.values()), default=0)
                peak_worker, peak_tree = max(peak_worker, worker), max(peak_tree, tree)
                used = sum(path.stat().st_size for path in perf.OUTPUT.rglob("*") if path.is_file())
                disk = os.statvfs(perf.OUTPUT)
                available = disk.f_bavail * disk.f_frsize
                elapsed = time.monotonic() - clock
                samples.write(json.dumps(dict(epoch=time.time(), processes=current, worker_RSS=worker,
                    tree_RSS=tree, artifact_bytes=used, available_bytes=available, elapsed_seconds=elapsed)) + "\n")
                samples.flush()
                if worker > 96 * 2**30 or tree > 128 * 2**30:
                    violation = "RSS cap"
                elif used > 640 * 2**30 or available < 100 * 2**30:
                    violation = "artifact/disk headroom cap"
                elif elapsed > 12 * 3600:
                    violation = "12h timeout"
                if violation:
                    monitor.terminate(process, current)
                    break
                time.sleep(1)
        status = process.wait()
    (OUTPUT / "runtime.exit").write_text(str(status) + "\n")
    write("runtime-terminal.json", dict(actual_child_exit=status, violation=violation,
        max_worker_RSS=peak_worker, max_tree_RSS=peak_tree, CPU="48-63", GPU=4, no_retry=True))
    raise SystemExit(1 if status or violation else 0)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("provider", "quiet", "snapshot", "runtime"), required=True)
    parser.add_argument("--role", choices=("SOURCE", "AFTER"))
    parser.add_argument("--label")
    args = parser.parse_args()
    source.scope()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if args.mode == "provider":
        provider(args.role)
    elif args.mode == "quiet":
        quiet(args.label)
    elif args.mode == "snapshot":
        write(args.label + "-process-snapshot.json", dict(active=active_maintenance(), GPU4_processes=active_gpu4(), stack=stable_stack()))
        assert not active_maintenance() and not active_gpu4()
    else:
        supervise()


if __name__ == "__main__":
    main()

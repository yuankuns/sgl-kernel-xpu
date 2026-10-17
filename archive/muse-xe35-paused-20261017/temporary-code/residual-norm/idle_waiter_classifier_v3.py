import hashlib
import json
from pathlib import Path
import subprocess
import time


SCRIPT = Path("/usr/share/unattended-upgrades/unattended-upgrade-shutdown")
SCRIPT_SHA = "8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0"
ARGV = ["/usr/bin/python3", str(SCRIPT), "--wait-for-signal"]
LAST = {}


def idle(previous, current):
    if previous is None:
        return False
    for record in (previous, current):
        if not (record["pid"] == 4463 and record["starttick"] == "4954" and record["ppid"] == 1
                and record["state"] == "S" and record["args"] == ARGV
                and record["script_SHA256"] == SCRIPT_SHA and record["children"] == []
                and record["wchan"] == "poll_schedule_timeout.constprop.0" and record["write_bytes"] == 0):
            return False
    return current["epoch"] > previous["epoch"] and current["CPUticks"] == previous["CPUticks"]


def observe(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    args = [value.decode() for value in (directory / "cmdline").read_bytes().split(b"\0") if value]
    details = subprocess.run(["sudo", "-n", "cat", str(directory / "wchan"), str(directory / "io")],
        check=True, capture_output=True, text=True).stdout
    wchan, separator, io = details.partition("rchar:")
    assert separator
    counters = dict(line.split(":", 1) for line in ("rchar:" + io).splitlines())
    return dict(pid=pid, starttick=stat[19], ppid=int(stat[1]), state=stat[0], args=args,
        script_SHA256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        children=(directory / "task" / str(pid) / "children").read_text().split(),
        wchan=wchan.strip(), write_bytes=int(counters["write_bytes"]),
        CPUticks=int(stat[11]) + int(stat[12]), epoch=time.time())


def observed_idle(pid, output):
    try:
        current = observe(pid)
        previous = LAST.get(pid)
        if previous is None:
            previous = current
            time.sleep(1)
            current = observe(pid)
        result = idle(previous, current)
        LAST[pid] = current
        with (output / "idle-waiter-observations.jsonl").open("a") as logfile:
            logfile.write(json.dumps(dict(previous=previous, current=current, observed_idle=result)) + "\n")
        return result
    except (OSError, ValueError, AssertionError, subprocess.SubprocessError):
        return False


def contracts(output):
    record = dict(pid=4463, starttick="4954", ppid=1, state="S", args=ARGV,
        script_SHA256=SCRIPT_SHA, children=[], wchan="poll_schedule_timeout.constprop.0",
        write_bytes=0, CPUticks=6, epoch=1)
    later = dict(record, epoch=2)
    tests = dict(positive_idle_unchanged=idle(record, later),
        positive_idle_later_observation=idle(later, dict(later, epoch=3)),
        negative_wrong_argv=not idle(record, dict(later, args=["/usr/bin/python3", str(SCRIPT)])),
        negative_active_subtree=not idle(record, dict(later, children=["12345"], write_bytes=4096)))
    assert all(tests.values())
    (output / "CPU-classifier-contracts.json").write_text(json.dumps(dict(tests=tests, actual_exit=0), indent=2) + "\n")

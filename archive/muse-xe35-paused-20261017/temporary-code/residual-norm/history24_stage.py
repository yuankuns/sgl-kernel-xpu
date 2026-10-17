import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import history24_accuracy as check


def processes(root_pid):
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            rss = next(line for line in (path / "status").read_text().splitlines() if line.startswith("VmRSS:"))
            records[int(path.name)] = dict(ppid=int(fields[1]), starttick=int(fields[19]), rss=int(rss.split()[1]) * 1024)
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
            pass
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, record in records.items() if record["ppid"] in selected}
        if selected == expanded:
            break
        selected = expanded
    return {pid: records[pid] for pid in selected if pid in records}


def terminate(process, current):
    for pid in sorted(current, reverse=True):
        try:
            now = processes(pid).get(pid)
            if now and now["starttick"] == current[pid]["starttick"]:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for pid, record in processes(process.pid).items():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    argv = [args.python, "-B", str(check.ROOT / "history24_accuracy.py"), "--mode", args.mode]
    check.write(args.phase + ".command.json", argv)
    with (check.OUTPUT / (args.phase + ".log")).open("w") as logfile:
        process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / (args.phase + ".pid")).write_text(str(process.pid) + "\n")
        violation = False
        peak_worker = peak_tree = 0
        with (check.OUTPUT / (args.phase + "-RSS.jsonl")).open("w") as samples:
            while process.poll() is None:
                current = processes(process.pid)
                worker = current.get(process.pid, {}).get("rss", 0)
                tree = sum(record["rss"] for record in current.values())
                peak_worker = max(peak_worker, worker)
                peak_tree = max(peak_tree, tree)
                samples.write(json.dumps(dict(epoch=time.time(), processes=current, worker_RSS=worker, tree_RSS=tree)) + "\n")
                samples.flush()
                if worker > 16 * 2**30 or tree > 24 * 2**30:
                    violation = True
                    terminate(process, current)
                    break
                time.sleep(0.5)
        status = process.wait()
    (check.OUTPUT / (args.phase + ".exit")).write_text(str(status) + "\n")
    check.write(args.phase + "-supervisor.json", dict(actual_child_returncode=status, resource_violation=violation,
                peak_worker_RSS=peak_worker, peak_descendant_tree_RSS=peak_tree, namespace="container PPID descendants, not PGID"))
    print(json.dumps(dict(phase=args.phase, actual_child_returncode=status, resource_violation=violation)), flush=True)
    if status or violation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

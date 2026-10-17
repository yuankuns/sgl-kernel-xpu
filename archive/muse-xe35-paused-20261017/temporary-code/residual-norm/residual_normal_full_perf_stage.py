import argparse
import json
import subprocess
import time

import history24_stage as monitor
import residual_normal_full_perf as check


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.command[1:] if args.command and args.command[0] == "--" else args.command
    command = [args.python, "-B", str(check.ROOT / "residual_normal_full_perf.py"), *arguments]
    check.write(args.phase + ".command.json", command)
    violation = False
    peak_worker = peak_tree = 0
    with (check.OUTPUT / (args.phase + ".log")).open("w") as logfile:
        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
        (check.OUTPUT / (args.phase + ".pid")).write_text(str(process.pid) + "\n")
        with (check.OUTPUT / (args.phase + "-RSS.jsonl")).open("w") as samples:
            while process.poll() is None:
                current = monitor.processes(process.pid)
                worker = current.get(process.pid, {}).get("rss", 0)
                tree = sum(record["rss"] for record in current.values())
                peak_worker, peak_tree = max(peak_worker, worker), max(peak_tree, tree)
                samples.write(json.dumps(dict(epoch=time.time(), processes=current, worker_RSS=worker, tree_RSS=tree)) + "\n")
                samples.flush()
                if worker > 96 * 2**30 or tree > 128 * 2**30:
                    violation = True
                    monitor.terminate(process, current)
                    break
                time.sleep(0.5)
        status = process.wait()
    (check.OUTPUT / (args.phase + ".exit")).write_text(str(status) + "\n")
    check.write(args.phase + "-supervisor.json", dict(actual_exit=status, resource_violation=violation,
        peak_worker_RSS=peak_worker, peak_tree_RSS=peak_tree, worker_cap_GiB=96, tree_cap_GiB=128,
        monitor="container recursive PPID/starttick; only owned descendants terminated"))
    raise SystemExit(1 if status or violation else 0)


if __name__ == "__main__":
    main()

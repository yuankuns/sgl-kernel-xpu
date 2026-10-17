import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def processes_xe35():
    rows = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,args"], text=True).splitlines()[1:]
    return {int(parts[0]): {"parent": int(parts[1]), "state": parts[2], "args": parts[3]}
            for row in rows if len(parts := row.strip().split(None, 3)) == 4}


def descendants_xe35(processes, root):
    descendants = [root]
    for parent in descendants:
        descendants.extend(pid for pid, process in processes.items() if process["parent"] == parent)
    return descendants


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pause", "resume"])
    args = parser.parse_args()
    state_path = Path(__file__).resolve().parent / "full_build_paused_pids.json"
    command = "bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/build_port_xe35.sh"
    if args.action == "pause":
        assert not state_path.exists(), "live build already paused"
        processes = processes_xe35()
        roots = [pid for pid, process in processes.items() if process["args"] == command]
        assert len(roots) == 1, roots
        saved = {}
        for _ in range(3):
            processes = processes_xe35()
            for pid in descendants_xe35(processes, roots[0]):
                if pid in processes:
                    saved[str(pid)] = processes[pid]
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
            time.sleep(0.3)
        current = processes_xe35()
        assert all("T" in current[int(pid)]["state"] for pid in saved if int(pid) in current)
        state_path.write_text(json.dumps(saved, indent=2) + "\n")
        print("paused_owned_build_processes", len(saved), "root", roots[0], flush=True)
    else:
        saved = json.loads(state_path.read_text())
        current = processes_xe35()
        resumed = 0
        for pid_string, process in reversed(list(saved.items())):
            pid = int(pid_string)
            if pid in current and current[pid]["args"] == process["args"]:
                os.kill(pid, signal.SIGCONT)
                resumed += 1
        state_path.rename(state_path.with_name("full_build_resumed_pids.json"))
        print("resumed_owned_build_processes", resumed, flush=True)


if __name__ == "__main__":
    main_xe35()

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from stop_full_build import descendants, processes


artifact = Path(__file__).resolve().parent
command = "bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/build_port_xe35.sh"
snapshot = processes()
roots = [pid for pid, entry in snapshot.items() if entry["args"] == command]
assert len(roots) == 1, roots
saved = {}
for attempt in range(4):
    snapshot = processes()
    for pid in descendants(snapshot, roots[0]):
        if pid in snapshot:
            saved[str(pid)] = snapshot[pid]
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
    time.sleep(0.3)
current = processes()
assert all("T" in current[int(pid)]["state"] for pid in saved if int(pid) in current)
for pid_string in reversed(list(saved)):
    try:
        os.kill(int(pid_string), signal.SIGTERM)
        os.kill(int(pid_string), signal.SIGCONT)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 40
remaining = {}
while time.monotonic() < deadline:
    current = processes()
    remaining = {pid: current[int(pid)] for pid, entry in saved.items()
                 if int(pid) in current and current[int(pid)]["args"] == entry["args"]
                 and "Z" not in current[int(pid)]["state"]}
    if not remaining:
        break
    time.sleep(0.5)
for pid_string in remaining:
    os.kill(int(pid_string), signal.SIGKILL)
if remaining:
    time.sleep(1)
current = processes()
active = {pid: current[int(pid)] for pid, entry in saved.items()
          if int(pid) in current and current[int(pid)]["args"] == entry["args"]
          and "Z" not in current[int(pid)]["state"]}
assert not active, active
subprocess.run(["flock", "-n", "/root/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35/.build-review.lock", "true"], check=True)
record = {"reason": "user-authorized canonical default-all build deduplication; no local restart",
          "root": roots[0], "stopped_processes": saved, "forced_kill": remaining,
          "active_survivors": active, "build_tree_lock_released": True,
          "source_diagnostic_gpu7_untouched": True, "time_epoch": time.time()}
(artifact / "full_build_stop_canonical.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"Stopped {len(saved)} owned build processes; forced kills={len(remaining)}; active survivors=0; tree lock released", flush=True)

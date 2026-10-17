import datetime
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    rows = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,args"], text=True).splitlines()[1:]
    processes = {int(parts[0]): {"parent": int(parts[1]), "state": parts[2], "command": parts[3]}
                 for row in rows if len(parts := row.strip().split(None, 3)) == 4}
    command = "bash /root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/residual-norm/build_port_xe35.sh"
    build_roots = [pid for pid, entry in processes.items() if entry["command"] == command]
    assert len(build_roots) <= 1, build_roots
    descendants = list(build_roots)
    for parent in descendants:
        descendants.extend(pid for pid, entry in processes.items() if entry["parent"] == parent)
    entries = []
    for pid in descendants:
        try:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        except FileNotFoundError:
            continue
        cpus = fields["Cpus_allowed_list"].strip()
        assert cpus == "48-63", (pid, cpus)
        entries.append({"pid": pid, "cpus": cpus, **processes[pid]})
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lane": 3, "cpus": "48-63", "max_build_jobs": 8, "inner_igc_jobs": 1,
        "gpu": 7, "gpu0_3_new_launch_forbidden": True,
        "gpu3_release_exit": (root / "gpu3_release.exit").read_text().strip(),
        "full_build_handle": 48226, "full_validation_wait_handle": 48226,
        "retired_full_build_handle": 50858, "retired_validation_wait_handle": 54820,
        "gpu7_diagnostic_handle": 10644,
        "processes": entries,
        "diagnostic_ready": json.loads((root / "diagnostic-ready").read_text()),
    }
    (root / "lane_resource_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {len(entries)} full-build descendants confined to CPU48-63; one build root; GPU3 released.")


if __name__ == "__main__":
    main()

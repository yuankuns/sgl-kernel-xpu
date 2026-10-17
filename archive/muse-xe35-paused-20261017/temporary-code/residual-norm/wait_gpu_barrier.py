import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["diagnostic", "official", "long-graph"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    default_barrier = {"diagnostic": "gpu7-barrier.json", "official": "gpu7-official-barrier.json",
                       "long-graph": "gpu7-long-graph-barrier.json"}[args.phase]
    barrier = Path(os.environ.get("MUSE_BARRIER_FILE", root / default_barrier))
    ready = {
        "gpu": 7, "cpu_lane": 3, "cpus": "48-63", "phase": args.phase,
        "scope": "B8/ISL4096/OSL16/TP1/EP1", "barrier": str(barrier),
        "status": "waiting for main GPU4-11 coordinator release; no GPU lock held",
        "gpu3_released": True,
    }
    (root / f"{args.phase}-ready").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(ready), flush=True)
    epoch = os.environ.get("MUSE_SYNC_START_EPOCH")
    if epoch is None:
        while not barrier.exists():
            time.sleep(1)
        record = json.loads(barrier.read_text())
        assert record["gpu"] == 7 and record["coordinator_release"] is True
        epoch = record["start_epoch"]
    deadline = float(epoch)
    assert deadline > 0
    print(f"GPU7 {args.phase} main GPU4-11 start epoch: {deadline}", flush=True)
    time.sleep(max(0, deadline - time.time()))


if __name__ == "__main__":
    main()

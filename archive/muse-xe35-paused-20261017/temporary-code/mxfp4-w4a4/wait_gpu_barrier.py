import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=("mxfp4-w4a4", "routing-quant"), required=True)
    parser.add_argument("--phase", choices=("diagnostic", "official"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / args.module
    filename = "gpu8-barrier.json" if args.phase == "diagnostic" else "gpu8-official-barrier.json"
    barrier = Path(os.environ.get("MUSE_BARRIER_FILE", root / filename))
    ready = {
        "gpu": 8, "cpu_lane": 4, "cpus": "64-79", "module": args.module,
        "phase": args.phase, "scope": "B8/ISL4096/OSL16/TP1/EP1",
        "barrier": str(barrier), "gpu4_released": True,
        "status": "waiting main GPU4-11 release; no GPU lock held",
    }
    (root / (args.phase + "-barrier-ready.json")).write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(ready), flush=True)
    epoch = os.environ.get("MUSE_SYNC_START_EPOCH")
    if epoch is None:
        while not barrier.exists():
            time.sleep(1)
        record = json.loads(barrier.read_text())
        if record["gpu"] != 8 or record["coordinator_release"] is not True:
            raise RuntimeError("invalid main GPU8 barrier release")
        epoch = record["start_epoch"]
    deadline = float(epoch)
    if not math.isfinite(deadline) or deadline <= 0:
        raise RuntimeError("invalid main barrier start epoch")
    print(f"GPU8 {args.module} {args.phase} main GPU4-11 epoch: {deadline}", flush=True)
    time.sleep(max(0, deadline - time.time()))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import torch


def cases_xe35(path):
    cases = {}
    for line in path.read_text().splitlines():
        if line.startswith('{"provider"'):
            case = json.loads(line)
            key = (case["rows"], case["post_mode"], case["round_pre_norm_input"], case["output_fp32"])
            cases[key] = case
    return cases


def main_xe35():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-log", default="port_accuracy.log")
    parser.add_argument("--port-outputs", default="port-outputs")
    args = parser.parse_args()
    source = cases_xe35(root / "source_accuracy.log")
    port = cases_xe35(root / args.port_log)
    assert len(source) == len(port) == 24
    assert source.keys() == port.keys()
    for key, source_case in source.items():
        port_case = port[key]
        exact = all(source_case["accuracy"][name] == port_case["accuracy"][name]
                    for name in ("norm_sha256", "stream_sha256"))
        if not exact:
            filename = f"{key[0]}_{key[1]}_{int(key[2])}_{int(key[3])}.pt"
            source_outputs = torch.load(root / "source-outputs" / filename, map_location="cpu")
            port_outputs = torch.load(root / args.port_outputs / filename, map_location="cpu")
            torch.testing.assert_close(port_outputs["norm"].float(), source_outputs["norm"].float(),
                                       rtol=5e-3, atol=5e-3)
            torch.testing.assert_close(port_outputs["stream"], source_outputs["stream"],
                                       rtol=1e-5, atol=1e-5)
        print(json.dumps({"case": key, "bit_exact_source_to_port": exact,
                          "source": source_case["accuracy"], "port": port_case["accuracy"]}), flush=True)


if __name__ == "__main__":
    main_xe35()

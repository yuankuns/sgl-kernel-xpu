import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--phase", choices=("performance",), required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--calls", type=int, default=128)
    args = parser.parse_args()
    for name in ("BATCHES", "BATCH", "SERVING_BATCH", "BATCH_SIZE", "SGLANG_BATCH_SIZE", "SGLANG_MUSE_BATCH_SIZE", "MUSE_BATCH_SIZE", "SERVING_BATCH_SIZE", "MUSE_BS", "BS"):
        if os.environ.get(name, "8") != "8":
            parser.error(f"B8-only: {name}")
    for name, expected in {"ISL": "4096", "OSL": "16", "TP": "1", "EP": "1"}.items():
        if os.environ.get(name, expected) != expected:
            parser.error(f"Wrong production scope: {name}")
    gpu = "11" if args.phase == "accuracy" else "7"
    if os.environ.get("ZE_AFFINITY_MASK") != gpu:
        parser.error(f"Wrong main allocation: {gpu} required")
    grant = json.loads(args.grant.read_text())
    for key, value in {"scope": 'logits-model-performance-v3', "phase": args.phase, "gpu": int(gpu), "serving_batch": 8, "isl": 4096, "osl": 16, "tp": 1, "ep": 1, "coordinator_release": True, "accuracy_authorized": args.phase == "accuracy", "performance_authorized": args.phase == "performance"}.items():
        if type(grant.get(key)) is not type(value) or grant[key] != value:
            parser.error(f"Wrong coordinator grant: {key}")
    if grant["driver_sha256"] != digest(Path(__file__)):
        parser.error("Changed immutable driver")
    if args.samples != 101 or args.calls != 128:
        parser.error("Fixed comparable performance method required")
    if not math.isfinite(grant["start_epoch"]) or grant["start_epoch"] <= 0:
        parser.error("Invalid start epoch")
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Refusing to overwrite evidence")
    if grant.get("graph_helper_sha256") != digest(ROOT / "simple_verified_graph_v1.py"):
        parser.error("Changed verified graph helper")
    time.sleep(max(0, grant["start_epoch"] - time.time()))
    args.grant_data = grant
    return args


def main():
    args = arguments()
    provenance = json.loads((ROOT / "SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json").read_text())
    config_path = Path(provenance["model_config"].replace("/home/syk/", "/root/", 1))
    assert digest(config_path) == provenance["model_config_sha256"] == args.grant_data["model_config_sha256"]
    config = json.loads(config_path.read_text())
    vocab, hidden = config["vocab_size"], config["hidden_size"]
    assert (vocab, hidden) == (202048, 8192)
    official = ROOT / "logits-vocab/canonical-v4-icpx"
    comparator_path = ROOT / "logits-vocab/canonical-validation-v4-icpx/source-comparator-manifest.json"
    assert digest(official / "source-binary-manifest.json") == args.grant_data["package_manifest_sha256"]
    assert digest(comparator_path) == args.grant_data["source_comparator_manifest_sha256"]
    comparator = json.loads(comparator_path.read_text())
    official_manifest = json.loads((official / "source-binary-manifest.json").read_text())
    assert digest(Path(comparator["source_native_manifest"])) == comparator["source_native_manifest_sha256"] == args.grant_data["source_native_manifest_sha256"]
    stage = official / "site" if args.label == "after" else Path(comparator["stage"])
    import sgl_kernel
    import torch

    assert Path(sgl_kernel.__file__).resolve() == stage / "sgl_kernel/__init__.py"
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch()
    expected_files = official_manifest["installed_files"] | official_manifest["python_snapshot_files"] if args.label == "after" else comparator["files"]
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/common_ops" in line or "libsgl-ops-sycl-" in line}
    mapped_sha256 = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage).as_posix()
        assert digest(path) == expected_files[relative]["sha256"]
        mapped_sha256[relative] = digest(path)
    assert "sgl_kernel/libsgl-ops-sycl-VocabMask.so" in mapped_sha256
    operations = {name: getattr(torch.ops.sgl_kernel, name + "_xe35") for name in ("vocab_parallel_embedding", "vocab_mask")}
    torch.manual_seed(42)
    weight_cpu = torch.randn((vocab, hidden), dtype=torch.bfloat16)
    weight = weight_cpu.to("xpu")
    weight_sha256 = hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    from simple_verified_graph_v1 import time_graph

    results, outputs = [], {}
    for index, (label, tokens) in enumerate((("decode8", 8), ("prefill-wave12288", 12288), ("prefill-wave20480", 20480))):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        ids_cpu = torch.randint(0, vocab, (tokens,), dtype=torch.int64)
        ids_cpu[:8] = torch.tensor([-1, 0, 1, vocab - 1, vocab, 100, 70000, 150000])
        ids = ids_cpu.to("xpu")
        output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="xpu")
        invalid = (ids_cpu < 0) | (ids_cpu >= vocab)
        expected = weight_cpu[ids_cpu.clamp(0, vocab - 1)].clone()
        expected[invalid] = 0
        launch = lambda: operations["vocab_parallel_embedding"](ids, weight, output, 0, vocab, 0, vocab, vocab)
        launch()
        torch.xpu.synchronize()
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        record = {"case": "embedding-" + label, "tokens": tokens, "hidden": hidden, "dtype": "bf16", "accuracy": "PASS", "full_reference_elements": actual.numel(), "input_ids_sha256": hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()}
        if args.phase == "performance":
            record.update(time_graph(torch, launch, args))
        outputs[record["case"]] = {"ids": ids_cpu, "output": actual}
        results.append(record)
        torch.xpu.synchronize()
        time.sleep(2)
        mask_launch = lambda: operations["vocab_mask"](ids, 0, vocab, 0, vocab, vocab)
        masked, mask = mask_launch()
        torch.testing.assert_close(masked.cpu(), ids_cpu * (~invalid), atol=0, rtol=0)
        torch.testing.assert_close(mask.cpu(), invalid, atol=0, rtol=0)
        outputs["mask-" + label] = {"ids": ids_cpu, "mapped": masked.cpu(), "invalid": mask.cpu()}
        mask_record = {"case": "mask-" + label, "tokens": tokens, "accuracy": "PASS", "reference": "full exact TP1 index and invalid mask"}
        if args.phase == "performance":
            mask_record.update(time_graph(torch, mask_launch, args))
        results.append(mask_record)
        print(json.dumps(record), flush=True)
    assert not args.output.exists()
    torch.save(outputs, args.output.with_suffix(".pt"))
    report = {"label": args.label, "phase": args.phase, "gpu": os.environ["ZE_AFFINITY_MASK"], "serving_batch": 8, "isl": 4096, "osl": 16, "tp": 1, "ep": 1, "weight_sha256": weight_sha256, "model_config_sha256": digest(config_path), "driver_sha256": digest(Path(__file__)), "mapped_native_sha256": mapped_sha256, "results": results, "raw_output_file": str(args.output.with_suffix(".pt")), "scope": "actualL3 config/call signature matching synthetic inputs, derived historical B8 wave lengths; invalid IDs are boundary sentinels; direct genuine registered kernel contract, not whole model or raw trace replay", "softcap": "retained separately as standalone operator, not enabled by this config", "samples": args.samples if args.phase == "performance" else 0, "calls": args.calls if args.phase == "performance" else 0}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

import argparse
import json
import pathlib
import re
import runpy
import statistics
import time


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--isl", type=int, choices=(4096,), default=4096)
    parser.add_argument("--osl", type=int, choices=(16,), default=16)
    parser.add_argument("--tp", type=int, choices=(1,), default=1)
    parser.add_argument("--ep", type=int, choices=(1,), default=1)
    parser.add_argument("--implementation", choices=("source", "port"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--port-original-abi", action="store_true")
    parser.add_argument("--evidence-prefix", default="")
    parser.add_argument("--input-forensics", action="store_true")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    canonical = bool(re.fullmatch(r"canonical-(?:v[1-9][0-9]*(?:-icpx)?-)?(?:precision|performance)-", args.evidence_prefix))
    if not canonical and args.evidence_prefix not in ("", "diagnostic-", "fresh-diagnostic-", "compat-diagnostic-"):
        parser.error("Invalid evidence prefix")
    if canonical and (args.precision_only != args.evidence_prefix.endswith("precision-") or args.port_original_abi):
        parser.error("Canonical evidence requires normal APIs and the matching precision/performance mode")
    if args.precision_only and args.evidence_prefix != "compat-diagnostic-" and not (canonical and args.evidence_prefix.endswith("precision-")):
        parser.error("precision-only evidence requires a dedicated compat or canonical precision prefix")
    import torch
    import sgl_kernel

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("W4 production validation requires Xe3.5")
    torch.set_num_threads(16)
    artifact = pathlib.Path(__file__).resolve().parent
    reference = runpy.run_path("/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/benchmark/accuracy_reference.py")["verify_grouped_output"]
    op = getattr(sgl_kernel, "moe_grouped_mm_nt_xe35_mxfp4_w4a4" if args.implementation == "source" or args.port_original_abi else "moe_grouped_mm_nt_mxfp4_w4a4_xe35")
    print(json.dumps({"package": sgl_kernel.__file__, "op": op.__name__, "device": torch.xpu.get_device_name(), "scope": "B8/ISL4096/OSL16/TP1/EP1"}), flush=True)
    for allocated_rows, output_width in ((256, 8192), (256, 4096), (257, 4096), (262144, 4096), (262144, 8192)):
        torch.xpu.synchronize()
        time.sleep(2)
        torch.manual_seed(42)
        counts = [0] * 256
        if allocated_rows <= 257:
            for expert in torch.randperm(256)[:8 * 8].tolist():
                counts[expert] = 4
        else:
            counts = [allocated_rows // 256] * 256
        rows = torch.tensor(counts, dtype=torch.int32, device="xpu")
        activations = torch.randint(0, 256, (allocated_rows, 2048), dtype=torch.uint8, device="xpu")
        weights = torch.randint(0, 256, (256, output_width, 2048), dtype=torch.uint8, device="xpu")
        activation_scales = torch.randint(124, 130, (allocated_rows, 128), dtype=torch.uint8, device="xpu")
        weight_scales = torch.randint(124, 130, (256, 128, output_width), dtype=torch.uint8, device="xpu")
        output = torch.empty((allocated_rows, output_width), dtype=torch.bfloat16, device="xpu")
        shape = f"{allocated_rows}-{output_width}-4096"
        input_hashes = None
        if args.input_forensics:
            from input_forensics import tensor_sha256
            input_hashes = {name: tensor_sha256(tensor) for name, tensor in (("activations", activations), ("activation_scales", activation_scales), ("weights", weights), ("weight_scales", weight_scales), ("rows_per_expert", rows))}
            print(json.dumps({"shape_input_fingerprints": shape, "implementation": args.implementation, "round": args.round, "inputs": input_hashes}), flush=True)

        def launch_xe35():
            op(output, activations, activation_scales, weights, weight_scales, rows, output_width, 4096, 256)

        launch_xe35()
        torch.xpu.synchronize()
        accuracy = reference(output, activations, activation_scales, weights, weight_scales, counts, max_rows=4)
        selected = []
        offset = 0
        for count in counts:
            if count:
                selected.extend((torch.linspace(0, count - 1, min(count, 4)).long() + offset).tolist())
            offset += count
        sample = output[selected].cpu()
        forensic_points = None
        if args.input_forensics:
            from input_forensics import fp64_difference_points
            forensic = json.loads((artifact / "diagnostic-equivalence-forensics.json").read_text())
            examples = forensic["shapes"][shape]["comparisons"]["0-1"]["examples"]
            source_sample = torch.load(artifact / f"diagnostic-equivalence-0-source-{shape}.pt", map_location="cpu", weights_only=True)
            port_sample = torch.load(artifact / f"diagnostic-equivalence-1-port-{shape}.pt", map_location="cpu", weights_only=True)
            forensic_points = fp64_difference_points(output, activations, activation_scales, weights, weight_scales, counts, examples, source_sample, port_sample)
        saved = artifact / f"{args.evidence_prefix}equivalence-{args.round}-{args.implementation}-{shape}.pt"
        torch.save(sample, saved)
        for unused in range(0 if args.precision_only else 10):
            launch_xe35()
        torch.xpu.synchronize()
        timings = []
        for unused in range(0 if args.precision_only else args.iterations):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            launch_xe35()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        report = {"shape": shape, "implementation": args.implementation, "round": args.round, "samples_ms": timings, "median_ms": statistics.median(timings) if timings else None, "accuracy": accuracy, "output_sample": str(saved), "precision_only": args.precision_only}
        if args.evidence_prefix == "compat-diagnostic-" or canonical:
            from input_forensics import tensor_sha256
            report.update({"valid_output_hash": tensor_sha256(output[:sum(counts)]), "valid_output_rows": sum(counts), "unwritten_slack_rows": allocated_rows - sum(counts), "full_output_hash_scope": "Every column of every valid routed row, after the final launch; uninitialized allocation slack excluded"})
        if args.input_forensics:
            report.update({"input_hashes": input_hashes, "fp64_difference_points": forensic_points, "forensic_timing_limit": "Input hashing and selected FP64 reference sampling run outside timing; do not mix these samples with the original diagnostic round."})
        print(json.dumps(report), flush=True)
        del activations, weights, activation_scales, weight_scales, output, sample
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main_xe35()

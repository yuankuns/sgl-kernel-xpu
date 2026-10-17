import argparse
import hashlib
import json
import pathlib
import re
import runpy
import statistics
import time


def main():
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
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    canonical = bool(re.fullmatch(r"canonical-(?:v[1-9][0-9]*(?:-icpx)?-)?(?:precision|performance)-", args.evidence_prefix))
    if not canonical and args.evidence_prefix not in ("", "diagnostic-"):
        parser.error("Invalid evidence prefix")
    if canonical and (args.precision_only != args.evidence_prefix.endswith("precision-") or args.port_original_abi):
        parser.error("Canonical evidence requires normal APIs and the matching precision/performance mode")
    import torch
    import sgl_kernel

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError("Routing production validation requires Xe3.5")
    torch.set_num_threads(16)
    artifact = pathlib.Path(__file__).resolve().parent
    suffix = "_xe35" if args.implementation == "port" and not args.port_original_abi else ""
    route = getattr(sgl_kernel, "muse_spark_moe_routing" + suffix)
    quant = getattr(sgl_kernel, "sgl_per_token_group_quant_fp4" + suffix)
    gather = getattr(sgl_kernel, "sgl_padded_gather_per_token_group_quant_fp4" + suffix)
    quant_reference = runpy.run_path("/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014/tests/test_per_token_group_quant_mxfp4.py")["quantize_to_mxfp4"]
    print(json.dumps({"package": sgl_kernel.__file__, "scope": "B8/ISL4096/OSL16/TP1/EP1", "device": torch.xpu.get_device_name()}), flush=True)

    def samples(launch):
        if args.precision_only:
            return []
        for unused in range(10):
            launch()
        torch.xpu.synchronize()
        timings = []
        for unused in range(args.iterations):
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        return timings

    def report(shape, launch, accuracy):
        timings = samples(launch)
        print(json.dumps({"shape": shape, "implementation": args.implementation, "round": args.round, "median_ms": statistics.median(timings) if timings else None, "samples_ms": timings, "accuracy": accuracy, "precision_only": args.precision_only}), flush=True)

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()

    for tokens in (8, 32768):
        torch.xpu.synchronize()
        time.sleep(2)
        generator = torch.Generator().manual_seed(20261017 + tokens)
        logits_cpu = torch.randn(tokens, 256, dtype=torch.float32, generator=generator)
        bias_cpu = torch.randn(256, dtype=torch.float32, generator=generator)
        logits = logits_cpu.to("xpu")
        bias = bias_cpu.to("xpu")
        weights = torch.empty(tokens, 8, dtype=torch.float32, device="xpu")
        ids = torch.empty(tokens, 8, dtype=torch.int32, device="xpu")
        multiplier = 0.3535533905932738

        def launch_route():
            route(weights, ids, logits, bias, multiplier, 1.0, 1e-15)

        launch_route()
        torch.xpu.synchronize()
        scores = torch.sigmoid(logits * multiplier)
        ranking = scores + bias
        torch.testing.assert_close(ranking.gather(1, ids.long()), torch.topk(ranking, 8, dim=-1, sorted=True).values, rtol=0, atol=0)
        selected = scores.gather(1, ids.long())
        expected_weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-15)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=2.5e-4)
        saved = artifact / f"{args.evidence_prefix}route-output-{args.round}-{args.implementation}-{tokens}.pt"
        torch.save({"weights": weights.cpu(), "ids": ids.cpu()}, saved)
        report(f"route-{tokens}-256-8-fp32", launch_route, {"reference": "original production routing gate", "atol": 2.5e-4, "rtol": 0, "output": str(saved)})
        routed_ids = ids.cpu().flatten()
        sorted_positions = torch.argsort(routed_ids, stable=True)
        counts = torch.bincount(routed_ids.long(), minlength=256)
        padded = ((counts + 3) // 4) * 4
        allocated_rows = tokens * 8 + 3 * min(256, tokens * 8)
        row_map_cpu = torch.zeros(allocated_rows, dtype=torch.int32)
        scale_map_cpu = torch.zeros(allocated_rows, 2, dtype=torch.int32)
        active_rows = []
        source_offset = destination_offset = 0
        for count, padded_count in zip(counts.tolist(), padded.tolist()):
            if not count:
                continue
            source_rows = sorted_positions[source_offset:source_offset + count] // 8
            row_map_cpu[destination_offset:destination_offset + count] = source_rows.int()
            scale_map_cpu[destination_offset:destination_offset + padded_count, 0] = destination_offset
            scale_map_cpu[destination_offset:destination_offset + count, 1] = padded_count
            scale_map_cpu[destination_offset + count:destination_offset + padded_count, 1] = -padded_count
            active_rows.extend(range(destination_offset, destination_offset + count))
            source_offset += count
            destination_offset += padded_count
        used_rows = destination_offset
        row_map = row_map_cpu.to("xpu")
        scale_map = scale_map_cpu.to("xpu")
        hidden_cpu = torch.randn(tokens, 4096, generator=generator).to(torch.bfloat16)
        hidden = hidden_cpu.to("xpu")

        def validate_quant(output_q, output_s, expected_values):
            expected_q, row_scales = quant_reference(expected_values)
            expected_scales = torch.zeros(allocated_rows, 128, dtype=torch.uint8)
            offset = 0
            for padded_count in padded.tolist():
                if padded_count:
                    expected_scales.flatten()[offset * 128:(offset + padded_count) * 128] = row_scales[offset:offset + padded_count].T.contiguous().flatten()
                    offset += padded_count
            padding_rows = scale_map_cpu[:used_rows, 1] < 0
            offset = 0
            for padded_count in padded.tolist():
                if padded_count:
                    block = expected_scales.flatten()[offset * 128:(offset + padded_count) * 128].view(128, padded_count)
                    block[:, padding_rows[offset:offset + padded_count]] = 0
                    offset += padded_count
            torch.testing.assert_close(output_q[active_rows].cpu(), expected_q[active_rows], rtol=0, atol=0)
            actual_scales = output_s.flatten()[:used_rows * 128].cpu()
            torch.testing.assert_close(actual_scales, expected_scales.flatten()[:used_rows * 128], rtol=0, atol=0)
            return {"reference": "original CPU E2M1/UE8M0 quant reference", "atol": 0, "rtol": 0, "q_sha256": tensor_hash(output_q[active_rows]), "scale_sha256": tensor_hash(actual_scales), "groups": 128, "allocated_rows": allocated_rows, "used_rows": used_rows, "padding_rows": int(padding_rows.sum()), "slack_rows": allocated_rows - used_rows, "dtype": "bfloat16"}

        torch.xpu.synchronize()
        time.sleep(2)

        def launch_gather():
            return gather(hidden, row_map, scale_map=scale_map)

        output_q, output_s = launch_gather()
        accuracy = validate_quant(output_q, output_s, hidden_cpu.index_select(0, row_map_cpu.long()))
        report(f"gather-quant-map-{allocated_rows}-4096-bf16", launch_gather, accuracy)
        del output_q, output_s, hidden, hidden_cpu
        torch.xpu.synchronize()
        time.sleep(2)
        gate_up_cpu = torch.randn(allocated_rows, 8192, generator=generator).to(torch.bfloat16)
        gate_up = gate_up_cpu.to("xpu")
        activation = (torch.nn.functional.silu(gate_up_cpu[:, :4096].float()) * gate_up_cpu[:, 4096:].float()).to(torch.bfloat16)

        def launch_swiglu():
            return quant(gate_up, fuse_silu_and_mul=True, scale_map=scale_map)

        output_q, output_s = launch_swiglu()
        accuracy = validate_quant(output_q, output_s, activation)
        report(f"swiglu-quant-map-{allocated_rows}-4096-bf16", launch_swiglu, accuracy)
        del gate_up, gate_up_cpu, activation, output_q, output_s
        del row_map, scale_map, scores, ranking, expected_weights, weights, ids, logits, bias
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main()

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import statistics
import time

import torch


def serving_scope_xe35():
    expected = {"BATCHES": "8", "BATCH": "8", "BATCH_SIZE": "8", "SERVING_BATCH": "8", "ISL": "4096", "OSL": "16", "TP": "1", "EP": "1"}
    for name, value in expected.items():
        if name in os.environ and os.environ[name] != value:
            raise ValueError(f"{name} must be {value}")


def device_samples_xe35(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.xpu.synchronize()
    return [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]


def tensor_hash_xe35(tensor):
    digest = hashlib.sha256()
    for chunk in tensor.split(256, dim=0):
        digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def compare_outputs_xe35(actual, expected, mode):
    maximum = 0.0
    exact = True
    for actual_chunk, expected_chunk in zip(actual.split(1024), expected.split(1024)):
        if mode == "mxfp8":
            assert torch.equal(actual_chunk, expected_chunk), "MXFP8 source/port must be bitwise equal"
        torch.testing.assert_close(actual_chunk, expected_chunk, rtol=5e-3, atol=5e-3)
        maximum = max(maximum, (actual_chunk - expected_chunk).abs().max().item())
        exact = exact and torch.equal(actual_chunk, expected_chunk)
    tolerance = 0.0 if mode == "mxfp8" else 5e-3
    return {"max_abs": maximum, "exact": exact, "rtol": tolerance, "atol": tolerance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-batch", type=int, choices=(8,), default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tail-only", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    serving_scope_xe35()
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("invalid timing controls")

    import sgl_kernel
    import source_nvfp4_woq_xe35 as source
    from sgl_kernel import nvfp4_woq_bf16_grouped_mm_xe35, nvfp4_woq_mxfp8_scaled_grouped_mm_xe35, nvfp4_woq_prep_xe35

    target = pathlib.Path("/root/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35")
    spec = importlib.util.spec_from_file_location("independent_reference_xe35", target / "benchmark/accuracy_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(8)
    print(json.dumps({"package": sgl_kernel.__file__, "source_bridge": source.__file__, "torch": torch.__version__, "gpu": torch.xpu.get_device_name(0), "scope": "B8/ISL4096/OSL16/TP1/EP1", "order": "BAAB; A=source, B=port", "timer": "XPU profiling events; original source timer structure"}), flush=True)

    shapes = [(3, 11, 192, 64), (3, 65, 256, 128)] if args.tail_only else [(256, rows, 8192, inner) for inner in (8192, 4096) for rows in (64, 262144)]
    regressions = []
    for experts, rows, channels, inner in shapes:
        torch.xpu.synchronize()
        time.sleep(2)
        torch.manual_seed(20261017 + rows + inner)
        counts = [rows // experts] * experts
        for expert in torch.randperm(experts)[:rows % experts].tolist():
            counts[expert] += 1
        problems = torch.tensor([[count, channels, inner] for count in counts], device="xpu", dtype=torch.int32)
        offsets = torch.tensor([sum(counts[:expert]) for expert in range(experts)], device="xpu", dtype=torch.int32)
        weights = torch.randint(0, 256, (experts, channels, inner // 2), device="xpu", dtype=torch.uint8)
        scales = torch.randint(1, 9, (experts, channels, inner // 16), device="xpu", dtype=torch.int8).to(torch.float8_e4m3fn)
        globals_dq = torch.exp2(torch.arange(experts, device="xpu", dtype=torch.float32).remainder(6) - 3)
        multipliers, exponents = nvfp4_woq_prep_xe35(scales, globals_dq)
        source_multipliers = torch.empty_like(multipliers)
        source_exponents = torch.empty_like(exponents)
        source.prep_xe35(source_multipliers, source_exponents, scales, globals_dq)
        torch.xpu.synchronize()
        assert torch.equal(source_multipliers, multipliers)
        assert torch.equal(source_exponents, exponents)
        print(json.dumps({"scale_equivalence": "full bitwise equality", "shape": [experts, rows, channels, inner], "c_sha256": tensor_hash_xe35(multipliers), "sfmx_sha256": tensor_hash_xe35(exponents), "packed_sha256": tensor_hash_xe35(weights)}), flush=True)
        workspace = torch.empty(64 * 1024 * 1024, device="xpu", dtype=torch.uint8)
        modes = ("bf16", "mxfp8") if args.tail_only else ("bf16",)
        for mode in modes:
            torch.xpu.synchronize()
            time.sleep(2)
            dtype = torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn
            activations = torch.randint(-4, 5, (rows, inner), device="xpu", dtype=torch.int8).to(dtype)
            activation_scales = torch.full((rows, inner // 32), 127, device="xpu", dtype=torch.uint8) if mode == "mxfp8" else None
            output_source = torch.empty((rows, channels), device="xpu", dtype=torch.float32)
            output_port = torch.empty_like(output_source)
            if mode == "bf16":
                call_source = lambda: source.bf16_xe35(output_source, activations, weights, source_exponents, source_multipliers, problems, offsets, workspace)
                call_port = lambda: nvfp4_woq_bf16_grouped_mm_xe35(output_port, activations, weights, exponents, multipliers, problems, offsets, workspace)
            else:
                call_source = lambda: source.mxfp8_xe35(output_source, activations, weights, activation_scales, source_exponents, source_multipliers, problems, offsets, workspace)
                call_port = lambda: nvfp4_woq_mxfp8_scaled_grouped_mm_xe35(output_port, activations, weights, activation_scales, exponents, multipliers, problems, offsets, workspace)
            call_source()
            call_port()
            torch.xpu.synchronize()
            equivalence = compare_outputs_xe35(output_port, output_source, mode)
            for output in (output_source, output_port):
                reference.verify_grouped_output_xe35(output, activations, weights, multipliers, exponents, counts, mode, activation_scales, max_rows=3)
            print(json.dumps({"output_equivalence": equivalence, "shape": [experts, rows, channels, inner], "mode": mode, "source_output_sha256": tensor_hash_xe35(output_source), "port_output_sha256": tensor_hash_xe35(output_port)}), flush=True)
            rounds = []
            for round_index in range(0 if args.accuracy_only else args.rounds):
                samples = {"source": [], "port": []}
                for label in ("port", "source", "source", "port"):
                    torch.xpu.synchronize()
                    time.sleep(2)
                    measured = device_samples_xe35(call_source if label == "source" else call_port, args.warmup, args.iterations)
                    assert all(value > 0 for value in measured), measured
                    samples[label].extend(measured)
                    print(json.dumps({"phase": label, "round": round_index, "shape": [experts, rows, channels, inner], "mode": mode, "samples_us": measured}), flush=True)
                ratio = statistics.median(samples["port"]) / statistics.median(samples["source"])
                rounds.append(ratio)
                print(json.dumps({"round": round_index, "ratio_port_source": ratio, "samples_us": samples}), flush=True)
            regression = statistics.median(rounds) > 1.05 if rounds else None
            if regression:
                regressions.append({"shape": [experts, rows, channels, inner], "mode": mode, "ratios": rounds})
            print(json.dumps({"summary": True, "shape": [experts, rows, channels, inner], "mode": mode, "accuracy_only": args.accuracy_only, "round_ratios": rounds, "regression_gt_5pct": regression}), flush=True)
            del activations, activation_scales, output_source, output_port, call_source, call_port
        del weights, scales, globals_dq, source_multipliers, source_exponents, multipliers, exponents, workspace, problems, offsets
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    if regressions:
        raise RuntimeError(f"repeated >5% regressions require diagnosis, not acceptance: {regressions}")


if __name__ == "__main__":
    main()

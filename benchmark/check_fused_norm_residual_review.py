"""Check residual-norm outputs against an independent CPU PyTorch evaluation."""

import argparse
import json
import time

import torch

from bench_fused_norm_residual_review import (
    framework_triton,
    framework_triton_w16,
    framework_triton_w32,
    fused,
    inputs,
    main_sycl,
    main_torch,
    parse_shape,
)


def bf16_ordered(value):
    bits = value.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits & 0x8000 != 0, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[8])
    parser.add_argument("--hidden", type=int, nargs="+", default=[8192])
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument(
        "--post-mode", choices=["none", "weight", "input_scale"], default="none"
    )
    parser.add_argument("--round-pre-norm-input", action="store_true")
    parser.add_argument(
        "--providers",
        choices=[
            "fused", "main_torch", "main_sycl", "framework_triton",
            "framework_triton_w16", "framework_triton_w32",
        ],
        nargs="+",
        default=["fused", "main_torch"],
    )
    args = parser.parse_args()
    providers = {
        "fused": fused,
        "main_torch": main_torch,
        "main_sycl": main_sycl,
        "framework_triton": framework_triton,
        "framework_triton_w16": framework_triton_w16,
        "framework_triton_w32": framework_triton_w32,
    }
    shapes = args.shape or [(rows, hidden) for rows in args.rows for hidden in args.hidden]
    acceptance = {name: {"shapes_pass": 0, "shapes_fail": 0, "output_fail": 0, "stream_fail": 0} for name in args.providers}
    for rows, hidden in shapes:
        tensors = inputs(rows, hidden)
        selected = (
            [0, 1, rows // 2, rows - 1] if rows > 1024 else slice(None)
        )
        cpu_tensors = tuple(
            (t[selected] if t.dim() == 2 else t).cpu() for t in tensors[:6]
        ) + (None, None)
        expected_output, expected_stream, rounded_post = main_torch(args, cpu_tensors, return_post=True)
        upper = torch.nextafter(rounded_post, torch.full_like(rounded_post, float("inf")))
        lower = torch.nextafter(rounded_post, torch.full_like(rounded_post, float("-inf")))
        post_ulp = torch.maximum(
            upper.float() - rounded_post.float(),
            rounded_post.float() - lower.float(),
        )
        stream_bound = 1e-5 + cpu_tensors[5].abs() * post_ulp
        output_bound = 0.02 + 0.02 * expected_output.float().abs()
        values = {}
        for provider in args.providers:
            output, stream = providers[provider](args, tensors)
            output = output[selected].cpu()
            stream = stream[selected].cpu()
            values[provider] = (output, stream)
            ulps = (bf16_ordered(output) - bf16_ordered(expected_output)).abs()
            output_error = (output.float() - expected_output.float()).abs()
            stream_error = (stream - expected_stream).abs()
            output_bad = output_error > output_bound
            stream_bad = stream_error > stream_bound
            output_fail = int(output_bad.sum())
            stream_fail = int(stream_bad.sum())
            accepted = output_fail == 0 and stream_fail == 0
            acceptance[provider]["shapes_pass" if accepted else "shapes_fail"] += 1
            acceptance[provider]["output_fail"] += output_fail
            acceptance[provider]["stream_fail"] += stream_fail
            worst = int(ulps.reshape(-1).argmax())
            worst_output = int((output_error - output_bound).reshape(-1).argmax())
            worst_stream = int((stream_error - stream_bound).reshape(-1).argmax())
            print(
                json.dumps(
                    {
                        "provider": provider,
                        "rows": rows,
                        "hidden": hidden,
                        "checked_rows": 4 if rows > 1024 else rows,
                        "post_mode": args.post_mode,
                        "round_pre_norm_input": args.round_pre_norm_input,
                        "output_max_ulp": int(ulps.max()),
                        "output_over_2_ulp": int((ulps > 2).sum()),
                        "output_max_abs": float(output_error.max()),
                        "worst_ulp_expected": float(
                            expected_output.reshape(-1)[worst]
                        ),
                        "worst_ulp_actual": float(output.reshape(-1)[worst]),
                        "stream_max_abs": float(stream_error.max()),
                        "stream_mean_abs": float(stream_error.mean()),
                        "criterion": "output abs<=0.02+0.02*abs(CPU); stream abs<=1e-5+abs(beta)*post_BF16_ULP",
                        "output_fail_count": output_fail,
                        "stream_fail_count": stream_fail,
                        "accepted": accepted,
                        "output_max_excess": float((output_error - output_bound).max()),
                        "stream_max_excess": float((stream_error - stream_bound).max()),
                        "worst_output_expected": float(expected_output.reshape(-1)[worst_output]),
                        "worst_output_actual": float(output.reshape(-1)[worst_output]),
                        "worst_stream_expected": float(expected_stream.reshape(-1)[worst_stream]),
                        "worst_stream_actual": float(stream.reshape(-1)[worst_stream]),
                    }
                ),
                flush=True,
            )
        if "fused" in values:
            fused_output, fused_stream = values["fused"]
            for other in (
                "main_sycl", "framework_triton",
                "framework_triton_w16", "framework_triton_w32",
            ):
                if other not in values:
                    continue
                other_output, other_stream = values[other]
                pair_output_error = (fused_output.float() - other_output.float()).abs()
                pair_worst = int(pair_output_error.reshape(-1).argmax())
                print(
                    json.dumps(
                        {
                            "pair": f"fused-{other}",
                            "rows": rows,
                            "hidden": hidden,
                            "output_max_ulp": int(
                                (bf16_ordered(fused_output) - bf16_ordered(other_output)).abs().max()
                            ),
                            "output_max_abs": float(pair_output_error.max()),
                            "stream_max_abs": float((fused_stream - other_stream).abs().max()),
                            "worst_pair_cpu": float(expected_output.reshape(-1)[pair_worst]),
                            "worst_pair_fused": float(fused_output.reshape(-1)[pair_worst]),
                            "worst_pair_other": float(other_output.reshape(-1)[pair_worst]),
                        }
                    ),
                    flush=True,
                )
        torch.xpu.synchronize()
        time.sleep(2)
    print(json.dumps({"acceptance_summary": acceptance}), flush=True)
    if any(result["shapes_fail"] for result in acceptance.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

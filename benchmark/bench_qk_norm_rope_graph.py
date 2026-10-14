"""Device-event timing of repeated QK norm/RoPE XPU graph replay."""

import hashlib
import os
import statistics
import time
from pathlib import Path

import torch
from sgl_kernel import fused_qk_norm_rope


def main():
    torch.manual_seed(42)
    tokens = int(os.environ["QK_TOKENS"])
    repeats = int(os.environ.get("QK_GRAPH_REPEATS", "32"))
    samples = int(os.environ.get("QK_SAMPLES", "21"))
    warmup = int(os.environ.get("QK_WARMUP", "5"))
    nq = int(os.environ.get("QK_Q_HEADS", "128"))
    nk = int(os.environ.get("QK_K_HEADS", "16"))
    nv = int(os.environ.get("QK_TAIL_HEADS", "144"))
    hd = int(os.environ.get("QK_HEAD_DIM", "64"))
    scale = 43.7840518911 / (hd**0.5)
    qkv = torch.randn(tokens, (nq + nk + nv) * hd, device="xpu:0", dtype=torch.bfloat16)
    initial = qkv.clone()
    q_weight = torch.full((hd,), scale, device="xpu:0", dtype=torch.bfloat16)
    k_weight = torch.ones(hd, device="xpu:0", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="xpu:0", dtype=torch.int32) % 4096

    def fn():
        fused_qk_norm_rope(
            qkv,
            nq,
            nk,
            nv,
            hd,
            1e-5,
            q_weight,
            k_weight,
            500000.0,
            False,
            positions,
            rotary_dim=hd,
        )

    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    qkv.copy_(initial)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(repeats):
            fn()
    torch.xpu.synchronize()

    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libsgl-ops-sycl-FusedQKNormRope.so" in line
    }
    for path in sorted(paths):
        print(
            f"loaded_lib={path} sha256={hashlib.sha256(Path(path).read_bytes()).hexdigest()}",
            flush=True,
        )

    for _ in range(warmup):
        qkv.copy_(initial)
        graph.replay()
    torch.xpu.synchronize()
    values = []
    for _ in range(samples):
        qkv.copy_(initial)
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repeats)

    ordered = sorted(values)
    print(
        f"graph tokens={tokens} dtype=bf16 heads={nq},{nk},{nv} hd={hd} "
        f"repeats={repeats} samples={samples} warmup={warmup} "
        f"device={torch.xpu.get_device_name(0)}",
        flush=True,
    )
    print(
        f"tokens={tokens} raw_us={','.join(f'{x:.3f}' for x in values)} "
        f"p50_us={statistics.median(values):.3f} "
        f"mad_us={statistics.median(abs(x - statistics.median(values)) for x in values):.3f} "
        f"p20_us={ordered[int((samples - 1) * 0.2)]:.3f} "
        f"p80_us={ordered[int((samples - 1) * 0.8)]:.3f}",
        flush=True,
    )
    time.sleep(2)


if __name__ == "__main__":
    main()

"""Repeated device-event comparison of QK norm/RoPE implementations.

Run with an installed sgl_kernel package and the selected target library in
that package. QK_IMPL=sglang or muse_spark_unfused selects the measured path.
"""

import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sgl_kernel import fused_qk_norm_rope


def main():
    torch.manual_seed(42)
    impl = os.environ.get("QK_IMPL", "sglang")
    tokens_list = [
        int(x)
        for x in os.environ.get(
            "QK_TOKENS", "1,4,8,16,32,64,128,256,512,2048,32768"
        ).split(",")
    ]
    samples = int(os.environ.get("QK_SAMPLES", "7"))
    warmup = int(os.environ.get("QK_WARMUP", "5"))
    nq = int(os.environ.get("QK_Q_HEADS", "128"))
    nk = int(os.environ.get("QK_K_HEADS", "16"))
    hd = int(os.environ.get("QK_HEAD_DIM", "64"))
    # Muse's gated QKV projection carries V16 plus an untouched O-gate128.
    nv = int(os.environ.get("QK_TAIL_HEADS", "144"))
    eps, base = 1e-5, 500000.0
    scale = 43.7840518911 / (hd**0.5)
    device = "xpu:0"
    print(
        f"impl={impl} dtype=bf16 heads={nq},{nk},{nv} hd={hd} "
        f"eps={eps} base={base} samples={samples} warmup={warmup} "
        f"device={torch.xpu.get_device_name(0)}",
        flush=True,
    )
    if os.environ.get("QK_DIAG_LIB_PATH", "1") == "1":
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libsgl-ops-sycl-FusedQKNormRope.so" in line
        }
        if not paths:
            print("loaded_lib=not_mapped_at_import", flush=True)
        for path in sorted(paths):
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            print(f"loaded_lib={path} sha256={digest}", flush=True)
    for tokens in tokens_list:
        qkv = torch.randn(
            tokens, (nq + nk + nv) * hd, device=device, dtype=torch.bfloat16
        )
        original_qkv = qkv.clone()
        q_weight = torch.full((hd,), scale, device=device, dtype=torch.bfloat16)
        k_weight = torch.ones(hd, device=device, dtype=torch.bfloat16)
        positions = torch.arange(tokens, device=device, dtype=torch.int32) % 4096
        if impl == "muse_spark_unfused":
            inv_freq = 1.0 / (
                base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
            )
            freqs = torch.outer(torch.arange(32768, dtype=torch.float32), inv_freq)
            cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
                device=device, dtype=torch.bfloat16
            )
            rope_positions = positions.to(torch.long)

            def fn():
                q = qkv[:, : nq * hd]
                k = qkv[:, nq * hd : (nq + nk) * hd]
                q = (
                    F.rms_norm(q.reshape(-1, hd), (hd,), None, eps)
                    .reshape_as(q)
                    .to(torch.bfloat16)
                    * scale
                )
                k = (
                    F.rms_norm(k.reshape(-1, hd), (hd,), None, eps)
                    .reshape_as(k)
                    .to(torch.bfloat16)
                )
                torch.ops.sgl_kernel.rotary_embedding(
                    rope_positions, q, k, hd, cache, False
                )

        elif impl == "sglang":

            def fn():
                fused_qk_norm_rope(
                    qkv,
                    nq,
                    nk,
                    nv,
                    hd,
                    eps,
                    q_weight,
                    k_weight,
                    base,
                    False,
                    positions,
                    rotary_dim=hd,
                )

        else:
            raise ValueError(impl)

        for _ in range(warmup):
            qkv.copy_(original_qkv)
            fn()
        torch.xpu.synchronize()
        values = []
        for _ in range(samples):
            qkv.copy_(original_qkv)
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000)
        print(
            f"tokens={tokens} raw_us={','.join(f'{x:.3f}' for x in values)} "
            f"p50_us={statistics.median(values):.3f} "
            f"mad_us={statistics.median(abs(x - statistics.median(values)) for x in values):.3f} "
            f"p20_us={sorted(values)[int((samples-1)*0.2)]:.3f} "
            f"p80_us={sorted(values)[int((samples-1)*0.8)]:.3f}",
            flush=True,
        )
        if os.environ.get("QK_PROFILE") == "1":
            qkv.copy_(original_qkv)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.XPU]
            ) as profiler:
                fn()
                torch.xpu.synchronize()
            with tempfile.TemporaryDirectory() as directory:
                trace = Path(directory) / "qk-trace.json"
                profiler.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
            names = [
                event.get("name", "")
                for event in events
                if event.get("cat") == "kernel"
            ]
            print(f"tokens={tokens} kernel_names={names}", flush=True)
        qkv = None
        original_qkv = None
        torch.xpu.synchronize()
        time.sleep(2)


if __name__ == "__main__":
    main()

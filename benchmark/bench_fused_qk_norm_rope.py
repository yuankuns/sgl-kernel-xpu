import itertools
import json
import os
import statistics
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import triton
from sgl_kernel import fused_qk_norm_rope


def profile_fused_qk_norm_rope(fn, samples=16):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as profiler:
        for _ in range(samples):
            fn()
        torch.xpu.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    names = sorted({event.get("name", "") for event in kernels})
    if len(kernels) != samples:
        raise RuntimeError(
            f"expected {samples} QK/RoPE kernel events, "
            f"found {len(kernels)}: {names}"
        )
    print(f"QK/RoPE profiler kernel names: {names}")
    durations = [event["dur"] / 1000 for event in kernels]
    return statistics.median(durations), min(durations), max(durations)


# Supported dtypes and their properties
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
}
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3fn": 1,
}


def llama_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def apply_rotary_emb_native(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Native PyTorch rotary embedding implementation.
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        is_neox_style: Whether to use Neox-style or interleaved style
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        # Neox style: split in half along head dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # Interleaved style: even and odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    head_dim: int,
    rotary_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    device: torch.device,
):
    """Compute inverse frequencies for YARN RoPE."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )

    if factor != 1.0:
        # YARN scaling
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)

        # Compute linear interpolation factor
        linear_func = (dim_range - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        ramp_func = torch.clamp(linear_func, 0.0, 1.0)

        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor

        inv_freq = (
            inv_freq_interpolation * (1.0 - ramp_func)
            + inv_freq_extrapolation * ramp_func
        )

    return inv_freq


def fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 1.0,
    high: float = 1.0,
    attention_factor: float = 1.0,
    rotary_dim: int = None,
) -> torch.Tensor:
    """
    Reference implementation in PyTorch for testing.

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Reshape QKV to separate Q, K, V
    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)

    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Apply RMSNorm to Q and K
    q_normalized = llama_rms_norm(q, q_weight, eps)
    k_normalized = llama_rms_norm(k, k_weight, eps)

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(
        head_dim, rotary_dim, base, factor, low, high, qkv.device
    )

    # Compute cos and sin for each position
    positions = position_ids.to(torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply attention factor
    cos = cos * attention_factor
    sin = sin * attention_factor

    # Apply RoPE to Q and K (only to rotary_dim portion)
    q_rot = q_normalized[..., :rotary_dim]
    q_pass = q_normalized[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_normalized[..., :rotary_dim]
    k_pass = k_normalized[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)

    return result


# Benchmark configurations
batch_size_range = [1, 2, 4, 8]
seq_len_range = [64, 128, 256, 512, 1024]
# DeepSeek-V3 config: 128 Q heads, 128 KV heads (MLA)
head_config_range = [
    (32, 8, 8, 128),  # Standard MQA config
    (32, 32, 32, 128),  # Standard MHA config
    (128, 128, 128, 128),  # DeepSeek-V3 style
]
is_neox_range = [True, False]
dtype_range = ["fp16", "bf16", "fp8_e4m3fn"]

configs = []
for batch_size, seq_len, (nq, nk, nv, hd), is_neox, dtype in itertools.product(
    batch_size_range, seq_len_range, head_config_range, is_neox_range, dtype_range
):
    configs.append((batch_size, seq_len, nq, nk, nv, hd, is_neox, dtype))

muse_spark_mode = os.environ.get("MUSE_SPARK", "0") == "1"
if muse_spark_mode:
    for name, required in (
        ("BATCHES", "8"),
        ("ISL", "4096"),
        ("OSL", "16"),
        ("TP", "1"),
        ("EP", "1"),
    ):
        if os.environ.get(name, required) != required:
            raise ValueError(f"Muse-Spark serving workload requires {name}={required}")
    configs = [
        (batch_size, seq_len, 128, 16, 16, 64, False, "bf16")
        for batch_size, seq_len in itertools.product([8], [4096])
    ]
default_providers = "muse_spark_unfused,sglang" if muse_spark_mode else "torch,sglang"
all_results = []
providers = [
    p.strip()
    for p in os.environ.get("PROVIDERS", default_providers).split(",")
    if p.strip()
]


# Support running the benchmark in "chunked" mode by setting environment variables:
# - NUM_CHUNKS: total number of chunks
# - CHUNK_IDX: index of this chunk (0-based)
num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
chunk_idx = int(os.environ.get("CHUNK_IDX", "0"))
# Support selecting explicit configs via env var SINGLE_CONFIG with CSV:
# "batch_size,seq_len,nq,nk,nv,head_dim,is_neox,dtype"
# Multiple configs may be given, separated by ";".
single_cfg = os.environ.get("SINGLE_CONFIG", "")
if single_cfg:
    selected = []
    for entry in single_cfg.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 8:
            raise RuntimeError(
                "SINGLE_CONFIG entries must have 8 comma-separated fields"
            )
        bsz = int(parts[0])

        sl = int(parts[1])
        nq = int(parts[2])
        nk = int(parts[3])
        nv = int(parts[4])
        hd = int(parts[5])
        is_neox_val = parts[6].lower() in ("1", "true", "yes")
        dtype_val = parts[7]
        config = (bsz, sl, nq, nk, nv, hd, is_neox_val, dtype_val)
        if muse_spark_mode and config != (8, 4096, 128, 16, 16, 64, False, "bf16"):
            raise ValueError(
                "Muse-Spark requires B8 ISL4096 OSL16 TP1 EP1 BF16 QK configuration"
            )
        selected.append(config)
    if not selected:
        raise RuntimeError("SINGLE_CONFIG was set but contained no config")
    configs = selected
    num_chunks = 1
    chunk_idx = 0
if num_chunks > 1:
    total = len(configs)
    chunk_size = (total + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, total)
    configs = configs[start:end]


def calculate_flops(
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rotary_dim: int,
) -> int:
    """
    Calculate FLOPs for fused QK norm + RoPE kernel.

    RMS Norm per head: ~2*head_dim (square + normalize + multiply)
    RoPE per head: ~6*rotary_dim (sin, cos, multiply, add operations)
    """
    # RMS norm: 2 FLOPs per element (square, rsqrt, multiply, scale)
    flops_rmsnorm_q = num_tokens * num_heads_q * head_dim * 4
    flops_rmsnorm_k = num_tokens * num_heads_k * head_dim * 4

    # RoPE: ~6 FLOPs per rotary dimension (sin, cos, 4 mul/add)
    flops_rope_q = num_tokens * num_heads_q * rotary_dim * 6
    flops_rope_k = num_tokens * num_heads_k * rotary_dim * 6

    total_flops = flops_rmsnorm_q + flops_rmsnorm_k + flops_rope_q + flops_rope_k

    return total_flops


def calculate_effective_bandwidth(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    time_ms: float,
    bytes_per_elem: int = 2,
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for fused QK norm + RoPE kernel.

    Memory: read/write QKV tensor + read weights
    """
    num_tokens = batch_size * seq_len
    # The V/tail portion determines the row stride but is not read or written.
    qk_bytes = num_tokens * (num_heads_q + num_heads_k) * head_dim * bytes_per_elem

    # Weight tensors
    weight_bytes = 2 * head_dim * bytes_per_elem  # q_weight + k_weight

    # Total bytes (read Q/K + write Q/K + read weights)
    total_bytes = 2 * qk_bytes + weight_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(
        num_tokens, num_heads_q, num_heads_k, head_dim, rotary_dim
    )
    gflops = (total_flops / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
            "dtype",
        ],
        x_vals=configs,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-")][
            : len(providers)
        ],
        ylabel="us",
        plot_name="fused-qk-norm-rope-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    seq_len,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    is_neox,
    dtype,
    provider,
):
    device = torch.device("xpu")
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v
    torch_dtype = DTYPE_MAP[dtype]
    is_fp8 = dtype == "fp8_e4m3fn"

    if is_fp8:
        # FP8 tensors: create in float32, clamp to representable range, convert
        qkv = (
            torch.randn(
                num_tokens, num_heads * head_dim, device=device, dtype=torch.float32
            )
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
        q_weight = (
            torch.randn(head_dim, device=device, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
        k_weight = (
            torch.randn(head_dim, device=device, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
    else:
        qkv = torch.randn(
            num_tokens, num_heads * head_dim, device=device, dtype=torch_dtype
        )
        q_weight = torch.randn(head_dim, device=device, dtype=torch_dtype)
        k_weight = torch.randn(head_dim, device=device, dtype=torch_dtype)

    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
        if muse_spark_mode
        else torch.arange(num_tokens, device=device, dtype=torch.int32)
    )

    if muse_spark_mode:
        if dtype != "bf16" or head_dim != 64 or is_neox:
            raise RuntimeError(
                "MUSE_SPARK=1 requires bf16, head_dim=64, and is_neox=False"
            )
        eps = 1e-5
        base = 500000.0
        scale_query_by = 43.7840518911 / (head_dim**0.5)
        q_weight.fill_(scale_query_by)
        k_weight.fill_(1.0)
    else:
        eps = 1e-6
        base = 10000.0
    rotary_dim = head_dim

    quantiles = [0.5, 0.2, 0.8]

    if provider == "muse_spark_unfused":
        if not muse_spark_mode:
            raise RuntimeError("provider=muse_spark_unfused requires MUSE_SPARK=1")

        # Exact pre-fusion Muse Spark path:
        #   _per_head_norm(Q/K) -> Q scalar scale
        #   -> RotaryEmbedding.forward_xpu()
        # head_dim=64 dispatches RoPE to torch.ops.sgl_kernel.rotary_embedding.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(32768, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
            device=device, dtype=torch_dtype
        )
        rope_positions = position_ids.to(torch.long)

        def muse_spark_unfused():
            q_size = num_heads_q * head_dim
            k_size = num_heads_k * head_dim
            q = qkv[:, :q_size]
            k = qkv[:, q_size : q_size + k_size]
            q = (
                F.rms_norm(q.reshape(-1, head_dim), (head_dim,), None, eps)
                .reshape_as(q)
                .to(torch_dtype)
                * scale_query_by
            )
            k = (
                F.rms_norm(k.reshape(-1, head_dim), (head_dim,), None, eps)
                .reshape_as(k)
                .to(torch_dtype)
            )
            torch.ops.sgl_kernel.rotary_embedding(
                rope_positions,
                q,
                k,
                head_dim,
                cos_sin_cache,
                is_neox,
            )

        fn = muse_spark_unfused
    elif provider == "torch_cached":
        inv_freq = compute_inv_freq_yarn(
            head_dim, rotary_dim, base, 1.0, 1.0, 1.0, device
        )
        freqs = torch.outer(position_ids.to(torch.float32), inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()

        def torch_cached():
            view = qkv.view(num_tokens, num_heads, head_dim)
            q = view[:, :num_heads_q, :]
            k = view[:, num_heads_q : num_heads_q + num_heads_k, :]
            q = apply_rotary_emb_native(
                llama_rms_norm(q, q_weight, eps), cos_cache, sin_cache, is_neox
            )
            k = apply_rotary_emb_native(
                llama_rms_norm(k, k_weight, eps), cos_cache, sin_cache, is_neox
            )
            view[:, :num_heads_q, :] = q
            view[:, num_heads_q : num_heads_q + num_heads_k, :] = k

        fn = torch_cached
    elif provider == "torch":
        if is_fp8:
            qkv_ref = qkv.to(torch.float32)
            q_weight_ref = q_weight.to(torch.float32)
            k_weight_ref = k_weight.to(torch.float32)
            fn = lambda: fused_qk_norm_rope_reference(
                qkv_ref,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                eps,
                q_weight_ref,
                k_weight_ref,
                base,
                is_neox,
                position_ids,
                rotary_dim=rotary_dim,
            ).to(torch_dtype)
        else:
            fn = lambda: fused_qk_norm_rope_reference(
                qkv,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                eps,
                q_weight,
                k_weight,
                base,
                is_neox,
                position_ids,
                rotary_dim=rotary_dim,
            )
    elif provider == "sglang":
        fn = lambda: fused_qk_norm_rope(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor=1.0,
            low=1.0,
            high=1.0,
            attention_factor=1.0,
            rotary_dim=rotary_dim,
        )

    if os.environ.get("MUSE_SPARK_PROFILE") == "1" and provider == "sglang":
        ms, min_ms, max_ms = profile_fused_qk_norm_rope(fn)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate effective bandwidth using the correct bytes-per-element for this dtype
    bw_metrics = calculate_effective_bandwidth(
        batch_size,
        seq_len,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        ms,
        bytes_per_elem=DTYPE_BYTES[dtype],
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": bw_metrics["num_tokens"],
            "num_heads_q": num_heads_q,
            "num_heads_k": num_heads_k,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
            "is_neox": is_neox,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark.run(print_data=True)

    # Ensure results dir exists and write per-chunk CSV
    os.makedirs("benchmark/results", exist_ok=True)
    df = pd.DataFrame(all_results)
    chunk_label = (
        f"chunk_{os.environ.get('CHUNK_IDX','0')}_of_{os.environ.get('NUM_CHUNKS','1')}"
    )
    out_csv = os.path.join("benchmark/results", f"results_{chunk_label}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote results CSV: {out_csv}")

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)

    print(df.to_markdown(index=False))

    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby(["dtype", "provider"]).agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

    # Print best speedup
    print("\n" + "=" * 80)
    print("Speedup Analysis")
    print("=" * 80)

    # Pivot to compare providers
    pivot = df.pivot_table(
        index=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
            "dtype",
        ],
        columns="provider",
        values="time_us",
    )

    for baseline in ("torch", "torch_cached", "muse_spark_unfused"):
        if baseline not in pivot.columns or "sglang" not in pivot.columns:
            continue
        col = f"speedup_vs_{baseline}"
        pivot[col] = pivot[baseline] / pivot["sglang"]
        print(f"\nSpeedup vs {baseline}:")
        print(f"  overall average: {pivot[col].mean():.2f}x")
        print(f"  overall max:     {pivot[col].max():.2f}x")
        print(f"  overall min:     {pivot[col].min():.2f}x")

        # Per-dtype speedup breakdown
        for dt in df["dtype"].unique():
            mask = pivot.index.get_level_values("dtype") == dt
            sp = pivot.loc[mask, col]
            if not sp.empty:
                print(
                    f"  {dt:>12s}: avg={sp.mean():.2f}x  max={sp.max():.2f}x  min={sp.min():.2f}x"
                )
    print()
    print(pivot.to_markdown())
    print(f"Wrote results CSV: {out_csv}")

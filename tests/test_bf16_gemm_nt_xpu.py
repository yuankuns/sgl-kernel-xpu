import pytest
import torch
from sgl_kernel import bf16_gemm_nt, pack_bf16_gemm_weight


@pytest.fixture(autouse=True)
def seed_inputs():
    torch.manual_seed(42)


@pytest.mark.parametrize("capability,expected", [((3, 5), True), ((2, 0), False)])
def test_bf16_gemm_nt_dispatch_capability_without_device_name(
    monkeypatch, capability, expected
):
    from sgl_kernel.gemm import _bf16_gemm_nt_use_cute

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: capability
    )

    def reject_device_name(index):
        raise AssertionError(
            "dispatch must not use a device-model performance whitelist"
        )

    monkeypatch.setattr(torch.xpu, "get_device_name", reject_device_name)
    assert (
        _bf16_gemm_nt_use_cute(torch.empty(8, 4096), 8, 8192, 4096, False) == expected
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 32),  # Generic small GEMM.
        (7, 257, 96),  # Generic non-divisible output tail.
        (8, 202048, 8192),  # Muse B8 decode: M8, vocabulary N202048, K8192.
        (
            8,
            8192,
            4096,
        ),  # Muse B8 MoE decompression: M8, hidden N8192, compressed K4096.
        (8, 4096, 8192),  # Muse B8 MoE compression: M8, compressed N4096, hidden K8192.
        (17, 4096, 8192),
        (64, 202048, 8192),  # Synthetic GEMM row stress, not a serving batch.
        (65, 8192, 4096),  # Generic tile-boundary stress, not a serving batch.
    ],
)
def test_bf16_gemm_nt(shape):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    m, n, k = shape
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = bf16_gemm_nt(a, b)
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


def test_bf16_gemm_nt_packed_weight():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((8, 96), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((257, 96), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)[:, : b.shape[0]]
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [8, 9, 17, 32, 33, 96, 97, 256, 257])
def test_bf16_gemm_nt_packed_dense_projection(m):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    # Muse hidden projection N8192/K8192; only M8 is B8 decode, others are tile stress.
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((8192, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [1, 7, 8, 9])
def test_bf16_gemm_nt_packed_qkv_projection(m):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    # Muse Q/K/V+output-gate N18432/K8192; M8 is B8 decode, M1/7/9 are boundaries.
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((18432, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [1, 8, 9, 16, 17, 33])
@pytest.mark.parametrize("n,k", [(512, 96), (4096, 256)])
def test_bf16_gemm_nt_packed_tiles(m, n, k):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt.default(actual, a, packed_b, 2)
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m,n,k", [(8, 512, 96), (17, 4096, 256)])
def test_bf16_gemm_nt_native_kernel(m, n, k):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt.default(actual, a, b, 0)
    expected = torch.mm(a.float().cpu(), b.float().cpu().T)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 8192, 4112),
        (9, 8192, 4112),
        (8, 16640, 4080),
        (9, 16640, 4080),
        (8, 18432, 4080),
        (9, 18432, 4080),
    ],
)
def test_bf16_gemm_nt_native_dispatch_boundary(m, n, k):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    torch.testing.assert_close(
        bf16_gemm_nt(a, b).float().cpu(),
        torch.mm(a.float().cpu(), b.float().cpu().T),
        rtol=5e-3,
        atol=5e-3,
    )


def test_bf16_gemm_nt_packed_logical_width():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    for m, n, k in [
        (8, 272, 96),
        (8, 16384, 8192),
        (8, 202032, 8192),
        (8, 202048, 8192),
        (8, 202064, 8192),
    ]:
        a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
        packed_b = pack_bf16_gemm_weight(b)
        expected_width = (n + 255) // 256 * 256
        if expected_width >= 16384 and expected_width & (expected_width - 1) == 0:
            expected_width += 256
        assert packed_b.shape == (k, expected_width)
        torch.testing.assert_close(packed_b[:, :n], b.T, rtol=0, atol=0)
        actual = bf16_gemm_nt(a, packed_b, packed_weight=True, output_n=n)
        assert actual.shape == (m, n)
        torch.testing.assert_close(
            actual.float().cpu(),
            torch.mm(a.float().cpu(), b.float().cpu().T),
            rtol=5e-3,
            atol=5e-3,
        )


def test_bf16_gemm_nt_output_n_validation():
    a = torch.ones((1, 16), dtype=torch.bfloat16)
    weight = torch.ones((16, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a packed weight"):
        bf16_gemm_nt(a, weight, output_n=16)
    with pytest.raises(ValueError, match="within its width"):
        bf16_gemm_nt(a, weight, packed_weight=True, output_n=17)


def test_bf16_gemm_nt_prefill_projection():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    # Muse B8 ISL4096 prefill: M32768, hidden projection N8192, K8192.
    activations = torch.randn((8 * 4096, 8192), dtype=torch.bfloat16, device="xpu")
    weights = torch.randn((8192, 8192), dtype=torch.bfloat16, device="xpu")
    actual = bf16_gemm_nt(activations, weights)
    selected = torch.tensor([0, 1, 4095, 4096, 8191, 16384, 32766, 32767])
    expected = activations[selected.to("xpu")].float().cpu() @ weights.float().cpu().T
    torch.testing.assert_close(
        actual[selected.to("xpu")].float().cpu(), expected, rtol=5e-3, atol=5e-3
    )

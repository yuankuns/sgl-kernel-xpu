
import pytest
import torch
from sgl_kernel import bf16_gemm_nt, pack_bf16_gemm_weight


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 32),
        (7, 257, 96),
        (8, 202048, 8192),
        (17, 4096, 8192),
        (64, 202048, 8192),
        (65, 8192, 4096),
    ],
)
def test_bf16_gemm_nt(shape):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    m, n, k = shape
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = bf16_gemm_nt(a, b)
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_bf16_gemm_nt_packed_weight():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((8, 96), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((257, 96), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)[:, : b.shape[0]]
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m", [8, 9, 17, 32, 33, 96, 97, 256, 257])
def test_bf16_gemm_nt_packed_dense_projection(m):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((8192, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m", [1, 7, 8, 9])
def test_bf16_gemm_nt_packed_qkv_projection(m):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((18432, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


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
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m,n,k", [(8, 512, 96), (17, 4096, 256)])
def test_bf16_gemm_nt_native_kernel(m, n, k):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt.default(actual, a, b, 0)
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


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
        bf16_gemm_nt(a, b), torch.mm(a, b.T), rtol=2e-2, atol=2e-2
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
        torch.testing.assert_close(actual, torch.mm(a, b.T), rtol=2e-2, atol=2e-2)


def test_bf16_gemm_nt_output_n_validation():
    a = torch.ones((1, 16), dtype=torch.bfloat16)
    weight = torch.ones((16, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a packed weight"):
        bf16_gemm_nt(a, weight, output_n=16)
    with pytest.raises(ValueError, match="within its width"):
        bf16_gemm_nt(a, weight, packed_weight=True, output_n=17)

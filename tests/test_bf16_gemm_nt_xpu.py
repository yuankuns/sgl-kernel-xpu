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
def test_bf16_gemm_nt_packed_muse_qkv_projection(m):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((18432, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight(b)
    actual = bf16_gemm_nt(a, packed_b, packed_weight=True)
    expected = torch.mm(a, b.T)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

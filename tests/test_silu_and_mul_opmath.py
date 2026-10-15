import pytest
import sgl_kernel
import torch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("columns", [252, 248, 256, 8192])
def test_silu_and_mul_cpu_fp32(dtype, columns):
    # Muse decode is B8/hidden8192; smaller widths cover generic vector edges.
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    generator = torch.Generator().manual_seed(20261016 + columns)
    values = (torch.randn(8, 2 * columns, generator=generator) * 3).to(dtype)
    expected = (
        torch.nn.functional.silu(values[:, :columns].float())
        * values[:, columns:].float()
    )
    actual = sgl_kernel.silu_and_mul(values.to("xpu"))
    torch.testing.assert_close(actual.cpu().float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("columns", [255, 254])
def test_silu_and_mul_rejects_unaligned_input(dtype, columns):
    values = torch.zeros(8, 2 * columns, dtype=dtype)
    assert values.shape[-1] * values.element_size() % 16 != 0
    with pytest.raises(
        ValueError, match=r"^The pointers must be multiple of 16 bytes\.$"
    ):
        sgl_kernel.silu_and_mul(values)


@pytest.mark.parametrize(
    "dtype,gate,up",
    [
        (torch.bfloat16, 0.5, 2.40625),
        (torch.float16, 0.004001617431640625, 374.0),
    ],
)
def test_silu_and_mul_cpu_fp32_rounding_boundary(dtype, gate, up):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    values = torch.zeros(8, 64, dtype=dtype)
    values[:, 0], values[:, 32] = gate, up
    values[:, 1], values[:, 33] = 4.0, 1.25
    expected = torch.nn.functional.silu(values[:, :32].float()) * values[:, 32:].float()
    actual = sgl_kernel.silu_and_mul(values.to("xpu"))
    torch.testing.assert_close(actual.cpu().float(), expected, atol=5e-3, rtol=5e-3)

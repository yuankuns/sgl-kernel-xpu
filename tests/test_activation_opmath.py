import pytest
import torch
from sgl_kernel import silu_and_mul


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,dim",
    [
        (1, 128),  # Generic activation corner.
        (8, 128),  # Generic small width, not a Muse production projection.
        (17, 256),  # Generic row/vector boundary.
        (64, 4096),  # Muse B8 topk8 routed SiLU: 64 rows, intermediate width4096.
        (256, 4096),  # Generic padded routed-row stress, not serving batch256.
    ],
)
def test_silu_mul_independent_fp32_math(dtype, rows, dim):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    generator = torch.Generator(device="cpu").manual_seed(0)
    inputs = (torch.randn(rows, 2 * dim, generator=generator) * 3).to(dtype)
    gate, up = inputs.float().chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    actual = silu_and_mul(inputs.to("xpu"))
    torch.testing.assert_close(actual.float().cpu(), expected, atol=5e-3, rtol=5e-3)

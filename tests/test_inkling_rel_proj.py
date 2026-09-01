import pytest
import torch
from sgl_kernel import rel_proj_small_t

try:
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False

pytestmark = pytest.mark.skipif(
    not HAS_XPU, reason="Inkling relative projection requires XPU"
)


def _make_packed_r(t: int, h: int, kv_heads: int, d: int) -> torch.Tensor:
    q_width = h * 128
    kv_width = 2 * kv_heads * 128
    r_width = h * d
    packed = torch.randn(
        t, q_width + kv_width + r_width, device="xpu", dtype=torch.bfloat16
    )
    return packed[:, q_width + kv_width :].view(t, h, d)


def _reference(
    r: torch.Tensor, proj: torch.Tensor, tau: torch.Tensor | None
) -> torch.Tensor:
    r_float = r.float()
    if tau is not None:
        r_float = (r_float * tau.view(-1, 1, 1)).bfloat16().float()
    return torch.einsum("thd,de->the", r_float, proj.float())


@pytest.mark.parametrize(
    ("t", "h", "kv_heads"),
    [
        (1, 24, 2),
        (9, 12, 1),
        (32, 6, 1),
    ],
)
def test_rel_proj_small_t_matches_inkling_shapes(t, h, kv_heads):
    d, e = 16, 1024
    r = _make_packed_r(t, h, kv_heads, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = 1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
    out = torch.empty(t, h, e, device="xpu", dtype=torch.bfloat16)

    returned = rel_proj_small_t(r, proj, tau, out)
    # The kernel accumulates in fp32 and rounds once, on store, to bf16, so the
    # reference has to be rounded the same way -- comparing a bf16 result to an
    # unrounded fp32 reference needs an atol as large as one bf16 mantissa step
    # at the largest output (1.6e-2 at |out| ~ 4), which is loose enough to hide
    # a real regression rather than to measure one.
    reference = _reference(r, proj, tau).bfloat16().float()

    assert returned.data_ptr() == out.data_ptr()
    assert out.is_contiguous()
    assert out.shape == (t, h, e)
    # rtol is two bf16 mantissa steps (bf16 keeps 8 fraction bits); atol only
    # covers the 16-term dot products that cancel to near zero, where the
    # kernel's accumulation order shows through at the fp32 level (worst case
    # measured over 9 seeds x 3 shapes: 2.9e-8).
    torch.testing.assert_close(out.float(), reference, rtol=2.0**-7, atol=1e-6)


def test_rel_proj_small_t_rejects_contiguous_r():
    t, h, d, e = 5, 12, 16, 1024
    r = torch.randn(t, h, d, device="xpu", dtype=torch.bfloat16)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16)
    tau = torch.ones(t, device="xpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="packed qkvr"):
        rel_proj_small_t(r, proj, tau)


@pytest.mark.parametrize(("d", "e"), [(13, 1024), (16, 65)])
def test_rel_proj_small_t_rejects_nonproduction_projection_shape(d, e):
    t, h = 5, 3
    packed = torch.randn(t, h * d + 19, device="xpu", dtype=torch.bfloat16)
    r = packed[:, : h * d].view(t, h, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16)
    tau = torch.ones(t, device="xpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="only production"):
        rel_proj_small_t(r, proj, tau)

import pytest
import torch
from sgl_kernel import rel_proj_small_t

try:
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False

pytestmark = pytest.mark.skipif(not HAS_XPU, reason="Inkling relative projection requires XPU")


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
    ("t", "h", "kv_heads", "with_tau"),
    [
        (1, 24, 2, True),
        (9, 12, 1, True),
        (32, 6, 1, True),
        (9, 12, 1, False),
    ],
)
def test_rel_proj_small_t_matches_inkling_shapes(t, h, kv_heads, with_tau):
    d, e = 16, 1024
    r = _make_packed_r(t, h, kv_heads, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = (
        1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
        if with_tau
        else None
    )
    out = torch.empty(t, h, e, device="xpu", dtype=torch.bfloat16)

    returned = rel_proj_small_t(r, proj, tau, out)
    reference = _reference(r, proj, tau)

    assert returned.data_ptr() == out.data_ptr()
    assert out.is_contiguous()
    assert out.shape == (t, h, e)
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("with_tau", [False, True])
def test_rel_proj_small_t_generic_tail_fallback(with_tau):
    t, h, d, e = 5, 3, 13, 65
    packed = torch.randn(t, h * d + 19, device="xpu", dtype=torch.bfloat16)
    r = packed[:, : h * d].view(t, h, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = (
        1.0 + 0.1 * torch.rand(t, device="xpu", dtype=torch.float32)
        if with_tau
        else None
    )

    out = rel_proj_small_t(r, proj, tau)
    reference = _reference(r, proj, tau)

    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


def test_rel_proj_small_t_tau_prescale_rounding():
    t, h, d, e = 9, 12, 16, 1024
    r = _make_packed_r(t, h, 1, d)
    proj = torch.randn(d, e, device="xpu", dtype=torch.bfloat16) * 0.1
    tau = 1.0 + 0.5 * torch.rand(t, device="xpu", dtype=torch.float32)

    out = rel_proj_small_t(r, proj, tau)
    pre_scaled_r = (r.float() * tau.view(-1, 1, 1)).bfloat16()

    assert torch.equal(out, rel_proj_small_t(pre_scaled_r.contiguous(), proj))

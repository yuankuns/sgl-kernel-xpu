import pytest
import sgl_kernel  # noqa: F401
import torch


def test_muse_softcap_logits_strided():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = torch.randn(3, 17, device="xpu", dtype=torch.float32)
    logits = source[:, :13]
    before = logits.clone()
    tail = source[:, 13:].clone()
    torch.ops.sgl_kernel.muse_softcap_logits(logits, 30.0)
    torch.testing.assert_close(logits, torch.tanh(before / 30.0) * 30.0)
    torch.testing.assert_close(source[:, 13:], tail)


@pytest.mark.parametrize("width", [32, 33])
def test_muse_softcap_logits_contiguous(width):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = torch.linspace(-90, 90, 4 * width, device="xpu").reshape(4, width)
    logits = source.clone()
    torch.ops.sgl_kernel.muse_softcap_logits(logits, 30.0)
    torch.testing.assert_close(logits, torch.tanh(source / 30.0) * 30.0)


def test_muse_softcap_logits_vector_accuracy_across_zero_and_saturation():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    source = torch.linspace(-900.0, 900.0, 2048, device="xpu", dtype=torch.float32)
    source[:8] = torch.tensor(
        [0.0, 1e-6, -1e-6, 0.001, -0.001, 2.99, 3.01, -3.01],
        device="xpu",
        dtype=torch.float32,
    )
    logits = source.clone().reshape(8, 256)
    torch.ops.sgl_kernel.muse_softcap_logits(logits, 30.0)
    torch.testing.assert_close(
        logits, (torch.tanh(source / 30.0) * 30.0).reshape(8, 256), rtol=3e-5, atol=3e-5
    )

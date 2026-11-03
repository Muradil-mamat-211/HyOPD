import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

# A source path allows testing the actual patched operator on minimal CPU hosts
# without importing unrelated optional GPU kernels from liger_kernel.__init__.
if os.getenv("LIGER_SCALED_CE_TEST_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "liger_scaled_ce_under_test", os.environ["LIGER_SCALED_CE_TEST_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Liger = module.LigerFusedLinearScaledCrossEntropyFunction
else:
    from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction as Liger


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("selected", ["full", "duplicates", "none"])
def test_outputs_and_all_gradients(device, temperature, entropy, selected):
    torch.manual_seed(42)
    h = torch.randn(7, 11, device=device, requires_grad=True)
    w = torch.randn(19, 11, device=device, requires_grad=True)
    y = torch.tensor([1, 3, -100, 0, 18, 8, 2], device=device)
    ids = {
        "full": torch.arange(19, device=device),
        "duplicates": torch.tensor([4, 2, 4, 18], device=device),
        "none": None,
    }[selected]
    out = Liger.apply(h, w, y, temperature, -100, 1, entropy, ids)
    out = out if isinstance(out, tuple) else (out,)
    logp = F.log_softmax(F.linear(h, w).float() / temperature, -1)
    valid = y != -100
    expected = [F.nll_loss(logp, y, reduction="none")]
    if entropy:
        expected.append((-(logp.exp() * logp).sum(-1)).masked_fill(~valid, 0))
    if ids is not None:
        expected.append(logp[:, ids].masked_fill(~valid[:, None], 0))
    scales = [torch.randn_like(t) for t in out]
    for actual, reference in zip(out, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    if selected == "full":
        torch.testing.assert_close(out[-1][valid].exp().sum(-1), torch.ones(6, device=device))
    actual_loss = sum((a * g).sum() for a, g in zip(out, scales, strict=True))
    expected_loss = sum((a * g).sum() for a, g in zip(expected, scales, strict=True))
    grads = torch.autograd.grad(actual_loss, (h, w))
    ref_grads = torch.autograd.grad(expected_loss, (h, w))
    for a, b in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    assert grads[0][2].count_nonzero() == 0


def test_teacher_no_grad_and_legacy_call():
    h = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(9, 4, requires_grad=True)
    y = torch.tensor([1, 2])
    with torch.inference_mode():
        outputs = Liger.apply(h, w, y, 1.0, -100, 1, True, torch.arange(9))
    assert all(not t.requires_grad for t in outputs)
    nll, entropy = Liger.apply(h, w, y, 1.0, -100, 1, True)
    (nll + entropy).sum().backward()
    assert h.grad is not None and w.grad is not None


def test_full_range_does_not_load_native_sm90_optional_dependencies(monkeypatch):

    def unexpected_resolution(device):
        raise AssertionError("Full-range output should not resolve optional native kernels")

    monkeypatch.setitem(Liger.apply.__globals__, "_resolve_implementation", unexpected_resolution)
    h, w = torch.randn(3, 4), torch.randn(7, 4)
    _, _, full = Liger.apply(h, w, torch.tensor([1, 2, 3]), 1.0, -100, 1, True, torch.arange(7))
    torch.testing.assert_close(full, (h @ w.T).log_softmax(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native BF16 selected kernel requires CUDA")
def test_cuda_bfloat16_selected_gradients():
    torch.manual_seed(19)
    h = (torch.randn(5, 16, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    w = (torch.randn(23, 16, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    labels = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    ids = torch.tensor([2, 0, 2, 22], device="cuda")
    nll, entropy, selected = Liger.apply(h, w, labels, 0.7, -100, 1, True, ids)
    ref = ((h @ w.T) / 0.7).float().log_softmax(-1)
    torch.testing.assert_close(selected, ref[:, ids], atol=0.02, rtol=0.02)
    reference = -ref.gather(1, labels[:, None]).sum() - (ref.exp() * ref).sum() + ref[:, ids].square().sum()
    grads = torch.autograd.grad(nll.sum() + entropy.sum() + selected.square().sum(), (h, w), retain_graph=True)
    reference_grads = torch.autograd.grad(reference, (h, w))
    for actual, expected in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.1)

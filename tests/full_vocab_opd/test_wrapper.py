import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


def load_wrapper():
    if os.getenv("LIGER_SCALED_CE_TEST_SOURCE"):
        spec = importlib.util.spec_from_file_location("patched_liger", os.environ["LIGER_SCALED_CE_TEST_SOURCE"])
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        ops = types.ModuleType("liger_kernel.ops")
        ops.LigerFusedLinearScaledCrossEntropyFunction = kernel.LigerFusedLinearScaledCrossEntropyFunction
        old = sys.modules.get("liger_kernel.ops")
        sys.modules["liger_kernel.ops"] = ops
        try:
            spec = importlib.util.spec_from_file_location(
                "wrapper", Path(__file__).parents[2] / "verl/utils/experimental/torch_functional.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if old is None:
                sys.modules.pop("liger_kernel.ops")
            else:
                sys.modules["liger_kernel.ops"] = old
        return module.FusedLinearForPPO
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    return FusedLinearForPPO


@pytest.mark.parametrize("shape", [(9, 7), (2, 5, 7)])
def test_wrapper_full_and_legacy(shape):
    cls = load_wrapper()
    torch.manual_seed(4)
    h = torch.randn(shape, requires_grad=True)
    w = torch.randn(13, 7, requires_grad=True)
    ids = torch.randint(13, shape[:-1])
    sampled, entropy, full = cls(impl_backend="liger")(h, w, ids, 0.7, return_full_log_probs=True)
    ref = ((h @ w.T) / 0.7).log_softmax(-1)
    torch.testing.assert_close(full, ref)
    torch.testing.assert_close(sampled, ref.gather(-1, ids[..., None]).squeeze(-1))
    grads = torch.autograd.grad(full.square().sum() + entropy.sum(), (h, w))
    assert all(g.count_nonzero() for g in grads)
    assert len(cls(impl_backend="liger")(h, w, ids)) == 2
    with pytest.raises(RuntimeError, match="patched Liger"):
        cls(impl_backend="torch")(h, w, ids, return_full_log_probs=True)

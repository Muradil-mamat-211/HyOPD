import math
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.distillation.coverage_adaptive_loss import TokenMetric, coverage_adaptive_full_vocab, coverage_metrics

SETTINGS = SimpleNamespace(top_p=0.8, coverage_ratio_threshold=0.2, coverage_threshold=0.8, residual_eps=1e-8)


def reference(p, logits, cfg):
    q = logits.softmax(-1)
    totals, rkl, rec, cov, ds = [], [], [], [], []
    for pp, qq in zip(p.double(), q.double(), strict=True):
        order = torch.argsort(pp, descending=True)
        keep, mass = [], 0.0
        for idx in order:
            keep.append(int(idx))
            mass += float(pp[idx])
            if mass >= cfg.top_p:
                break
        deficit = [i for i in keep if float((qq[i] / pp[i]).detach()) < cfg.coverage_ratio_threshold]
        c = sum(float(pp[i]) for i in keep if i not in deficit) / sum(float(pp[i]) for i in keep)
        a = min(c / cfg.coverage_threshold, 1.0)
        rr = (qq * (qq.log() - pp.log())).sum()
        if deficit:
            pm, qm = pp[deficit].sum(), qq[deficit].sum()
            ff = (pp[deficit] * (pp[deficit].log() - qq[deficit].log())).sum()
            ff += (1 - pm) * ((1 - pm).clamp_min(cfg.residual_eps).log() - (1 - qm).clamp_min(cfg.residual_eps).log())
        else:
            ff = qq.sum() * 0
        totals.append(a * rr + (1 - a) * ff)
        rkl.append(rr)
        rec.append(ff)
        cov.append(c)
        ds.append(deficit)
    return torch.stack(totals), torch.stack(rkl), torch.stack(rec), cov, ds


@pytest.mark.parametrize(
    "q",
    [
        [0.45, 0.28, 0.15, 0.12],
        [0.85, 0.01, 0.01, 0.13],
        [0.001, 0.001, 0.001, 0.997],
        [0.01, 0.8, 0.01, 0.18],
        [0.4, 0.01, 0.4, 0.19],
        [0.5, 0.3, 0.1, 0.1],
    ],
)
def test_router_rkl_recovery_and_gradient(q):
    p = torch.tensor([[0.50, 0.29, 0.11, 0.10]])
    logits = torch.tensor([q]).log().requires_grad_()
    actual, d = coverage_adaptive_full_vocab(p.log(), logits.log_softmax(-1), SETTINGS)
    expected, rkl, rec, c, deficit = reference(p, logits, SETTINGS)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(d["full_rkl"].double(), rkl, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(d["recovery"].double(), rec, atol=2e-6, rtol=2e-5)
    assert d["coverage"].item() == pytest.approx(c[0], abs=1e-6)
    assert d["deficit_mask"][0].nonzero().flatten().tolist() == deficit[0]
    assert d["alpha_r"].item() == pytest.approx(min(c[0] / 0.8, 1.0), abs=1e-6)
    for key in ("coverage", "alpha_r", "alpha_f", "deficit_mask"):
        assert not d[key].requires_grad
    g = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    gr = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(g, gr, atol=2e-6, rtol=2e-5)


def test_nucleus_includes_crossing_token_and_residual_is_present():
    p = torch.tensor([[0.51, 0.30, 0.10, 0.09]])
    q = torch.tensor([[0.97, 0.001, 0.001, 0.028]])
    _, d = coverage_adaptive_full_vocab(p.log(), q.log(), SETTINGS)
    assert d["nucleus_mask"].tolist() == [[True, True, False, False]]
    token_only = 0.30 * math.log(0.30 / 0.001)
    assert d["recovery"].item() != pytest.approx(token_only)
    assert d["recovery"].item() == pytest.approx(token_only + 0.7 * math.log(0.7 / 0.999), abs=1e-5)


def test_extreme_probabilities_and_empty_deficit():
    logits = torch.tensor([[100.0, -100.0, -1000.0], [-1000.0, 1000.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[100.0, -100.0, -1000.0], [1000.0, -1000.0, 0.0]]).log_softmax(-1)
    loss, d = coverage_adaptive_full_vocab(teacher, logits.log_softmax(-1), SETTINGS)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert d["deficit_size"][0] == 0 and d["recovery"][0] == 0


def test_token_metrics_merge_uneven_ranks_and_microbatches():
    a = TokenMetric(torch.tensor([0.0, 0.1]), quantile=0.5)
    b = TokenMetric(torch.tensor([0.7, 0.8, 0.9, 1.0]), quantile=0.5)
    merged = TokenMetric.aggregate_dp([a, b])
    assert merged.aggregate() == pytest.approx(0.75)
    mean = TokenMetric.aggregate_dp([TokenMetric(torch.tensor([0.0])), TokenMetric(torch.ones(9))])
    assert mean.aggregate() == pytest.approx(0.9)
    loss, details = coverage_adaptive_full_vocab(
        torch.randn(4, 9).log_softmax(-1), torch.randn(4, 9).log_softmax(-1), SETTINGS
    )
    assert len(coverage_metrics(details)) == 15


def test_residual_near_one_and_zero_action_rank():
    settings = SimpleNamespace(
        top_p=0.99999999, coverage_ratio_threshold=2.0, coverage_threshold=0.8, residual_eps=1e-8
    )
    logits = torch.tensor([[0.0, -80.0, -90.0]], requires_grad=True)
    log_p = torch.tensor([[0.0, -70.0, -100.0]]).log_softmax(-1)
    loss, _ = coverage_adaptive_full_vocab(log_p, logits.log_softmax(-1), settings)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()
    empty_logits = torch.empty(0, 9, requires_grad=True)
    empty, details = coverage_adaptive_full_vocab(empty_logits.detach(), empty_logits, SETTINGS)
    empty.sum().backward()
    assert empty_logits.grad is not None
    assert all(metric.aggregate() == 0 for metric in coverage_metrics(details).values())

# Copyright 2026 OpenAI
# Licensed under the Apache License, Version 2.0 (the "License").
"""Exact full-vocabulary reverse KL with coverage-adaptive residual recovery."""

import math

import numpy as np
import torch

from verl.utils.metric import AggregationType, Metric


@torch.no_grad()
def _coverage_routing(coverage, settings):
    """Return detached RKL/REC weights and the inclusive full-RKL route."""
    alpha_f = settings.alpha_max * ((settings.coverage_threshold - coverage) / settings.coverage_threshold).clamp(0, 1)
    alpha_r = 1 - alpha_f
    return alpha_r, alpha_f, coverage >= settings.coverage_threshold


def coverage_adaptive_full_vocab(log_p, log_q, settings):
    """Return one differentiable loss per action and detached routing diagnostics.

    Inputs are [actions, vocabulary]. Teacher quantization is corrected by a
    full-vocabulary FP32 normalizer; student logprobs retain their autograd graph.
    No top-k approximation or independent per-rank loss normalization is used.
    """
    if log_p.shape != log_q.shape or log_q.ndim != 2:
        raise ValueError(f"Expected matching [actions,vocabulary], got {log_p.shape} and {log_q.shape}")
    if log_q.shape[-1] == 0:
        raise ValueError("Full-vocabulary loss requires a nonempty vocabulary")
    log_q = log_q.float()
    with torch.no_grad():
        log_p = log_p.detach().float()
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        p = log_p.exp()
        sorted_p, order = p.sort(dim=-1, descending=True, stable=True)
        cumulative = sorted_p.cumsum(-1)
        # Include the first token that reaches/crosses top_p, then stop.
        nucleus_sorted = torch.cat(
            (torch.ones_like(cumulative[:, :1], dtype=torch.bool), cumulative[:, :-1] < settings.top_p), dim=-1
        )
        nucleus = torch.zeros_like(nucleus_sorted).scatter(1, order, nucleus_sorted)
        covered = log_q.detach() - log_p >= math.log(settings.coverage_ratio_threshold)
        deficit = nucleus & ~covered
        nucleus_mass = (p * nucleus).sum(-1)
        coverage = (p * (nucleus & covered)).sum(-1) / nucleus_mass
        alpha_r, alpha_f, full_rkl_route = _coverage_routing(coverage, settings)
        m_d = (p * deficit).sum(-1)
        # Sum the complement to avoid catastrophic cancellation for m_D ~ 1.
        residual_p = p.masked_fill(deficit, 0).sum(-1)

    q = log_q.exp()
    q_d = (q * deficit).sum(-1)
    full_rkl = (q * (log_q - log_p)).sum(-1)
    token_fkl = (p * (log_p - log_q)).masked_fill(~deficit, 0).sum(-1)
    residual_q = (1 - q_d).clamp_min(settings.residual_eps)
    residual = residual_p * (residual_p.clamp_min(settings.residual_eps).log() - residual_q.log())
    recovery = torch.where(deficit.any(-1), token_fkl + residual, torch.zeros_like(token_fkl))
    total = alpha_r * full_rkl + alpha_f * recovery
    details = {
        "coverage": coverage,
        "alpha_r": alpha_r,
        "alpha_f": alpha_f,
        "full_rkl_route": full_rkl_route,
        "nucleus_size": nucleus.sum(-1),
        "deficit_size": deficit.sum(-1),
        "deficit_teacher_mass": m_d,
        "deficit_student_mass": q_d.detach(),
        "full_rkl": full_rkl.detach(),
        "recovery": recovery.detach(),
        "total": total.detach(),
        "deficit_mask": deficit,
        "nucleus_mask": nucleus,
    }
    return total, details


class TokenMetric(Metric):
    """Merge token samples before means/quantiles across uneven DP microbatches.

    Only scalar per-action diagnostics cross the metrics channel, never logits.
    """

    def __init__(self, value=None, quantile=None):
        self.quantile = quantile
        self.aggregation = AggregationType.MEAN
        self.values = []
        if value is not None:
            self.append(value)

    def append(self, value):
        if isinstance(value, TokenMetric):
            self.extend(value)
        else:
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().numpy()
            self.values.append(np.asarray(value, dtype=np.float32).reshape(-1))

    def extend(self, values):
        for value in values.values if isinstance(values, TokenMetric) else values:
            self.append(value)

    def init_list(self):
        return TokenMetric(quantile=self.quantile)

    def aggregate(self):
        values = np.concatenate(self.values) if self.values else np.empty(0)
        if not len(values):
            return 0.0
        return float(np.mean(values) if self.quantile is None else np.quantile(values, self.quantile))

    @classmethod
    def aggregate_dp(cls, metrics):
        result = metrics[0].init_list()
        for metric in metrics:
            result.extend(metric)
        return result


def coverage_metrics(details):
    c = details["coverage"]
    metrics = {"coverage/mean": TokenMetric(c)}
    for name, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        metrics[f"coverage/{name}"] = TokenMetric(c, quantile=quantile)
    metrics["coverage/full_rkl_ratio"] = TokenMetric(details["full_rkl_route"])
    metrics["coverage/hybrid_ratio"] = TokenMetric(~details["full_rkl_route"])
    for key, name in (
        ("alpha_r", "alpha_r_mean"),
        ("alpha_f", "alpha_f_mean"),
        ("nucleus_size", "nucleus_size_mean"),
        ("deficit_size", "deficit_size_mean"),
        ("deficit_teacher_mass", "deficit_teacher_mass"),
        ("deficit_student_mass", "deficit_student_mass"),
    ):
        metrics[f"coverage/{name}"] = TokenMetric(details[key])
    for name in ("full_rkl", "recovery", "total"):
        metrics[f"loss/{name}"] = TokenMetric(details[name])
    return metrics

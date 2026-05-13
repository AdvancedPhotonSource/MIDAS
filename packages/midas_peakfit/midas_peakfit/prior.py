"""Aggregate log-prior over a parameter spec.

For Bayesian inference (Laplace / VI / HMC), the prior contribution is the
sum of ``Parameter.prior.log_prob(value)`` over all parameters that have a
declared prior.
"""
from __future__ import annotations

from typing import Dict

import torch

from midas_peakfit.spec import ParameterSpec


def sum_log_prior(unpacked: Dict[str, torch.Tensor], spec: ParameterSpec) -> torch.Tensor:
    """Return Σ log p(θ_i) over parameters that have a prior."""
    total = torch.zeros((), dtype=torch.float64)
    for name, param in spec.parameters.items():
        if param.prior is None:
            continue
        x = unpacked[name]
        try:
            total = total + param.prior.log_prob(x).sum()
        except Exception:
            continue
    return total


__all__ = ["sum_log_prior"]

"""Bound-constrained ↔ unconstrained parameter reparameterization.

LM optimizes in *unconstrained* ``u``-space; the model evaluates in
*bounded* ``x``-space. We use:

    x = lo + (hi - lo) × sigmoid(u)

This guarantees ``x ∈ [lo, hi]`` and is C¹-smooth, so autograd can
backprop through it. The inverse (used for initialization) is

    u = logit((x - lo) / (hi - lo))

with safe clamping near the edges to avoid ±inf.
"""
from __future__ import annotations

import torch

# Sigmoid saturates to 0 / 1 outside ~[-15, 15]; we clamp the inverse
# to keep gradients alive at the seed.
_LOGIT_EPS = 1e-6


def x_to_u(
    x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> torch.Tensor:
    """Inverse reparam: bounded → unconstrained.

    For seeds at exactly the bounds, clamps to ``[lo + ε, hi - ε]`` first.
    """
    span = hi - lo
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    t = (x - lo) / safe_span
    t = t.clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return torch.log(t / (1.0 - t))


def u_to_x(
    u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> torch.Tensor:
    """Forward reparam: unconstrained → bounded. Standard sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(u)


__all__ = ["x_to_u", "u_to_x"]

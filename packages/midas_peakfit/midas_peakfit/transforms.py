"""Bijective transforms between bounded parameter space and unconstrained space.

Inference (LM, LBFGS, Adam, VI, HMC) prefers an unconstrained parameter space.
We map every Parameter into an unconstrained representation via a
:class:`Transform`.  Each transform supplies:

  - ``forward(x) -> u``   bounded → unconstrained
  - ``inverse(u) -> x``   unconstrained → bounded
  - ``log_det_jacobian(u)`` for change-of-variable in the prior
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class Transform:
    """Abstract base.  Subclasses must implement forward / inverse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        """log |dx/du| evaluated at u, for change-of-variable in priors."""
        raise NotImplementedError


@dataclass
class Identity(Transform):
    """No-op (unbounded parameter)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u

    def log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=u.dtype, device=u.device)


@dataclass
class Log(Transform):
    """x = exp(u), for strictly positive parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp(min=1e-30))

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return torch.exp(u)

    def log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        # d(exp u)/du = exp u   →  log|J| = u
        return u.sum()


@dataclass
class Logit(Transform):
    """Sigmoid scaled to [lo, hi].  x = lo + (hi - lo) sigmoid(u)."""

    lo: float
    hi: float
    eps: float = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        span = self.hi - self.lo
        if span <= 0:
            raise ValueError(f"Logit transform requires hi > lo, got [{self.lo}, {self.hi}]")
        t = ((x - self.lo) / span).clamp(self.eps, 1.0 - self.eps)
        return torch.log(t / (1.0 - t))

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(u)

    def log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        # x = lo + span sigmoid(u)
        # dx/du = span sigmoid(u)(1 - sigmoid(u))
        s = torch.sigmoid(u)
        span = self.hi - self.lo
        return (torch.log(torch.tensor(span, dtype=u.dtype, device=u.device))
                + torch.log(s) + torch.log(1.0 - s)).sum()


@dataclass
class Scaled(Transform):
    """Affine: x = scale * u + offset.  Useful as a preconditioner."""

    scale: float
    offset: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return self.scale * u + self.offset

    def log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.tensor(abs(self.scale), dtype=u.dtype, device=u.device)) * u.numel()


__all__ = ["Transform", "Identity", "Log", "Logit", "Scaled"]

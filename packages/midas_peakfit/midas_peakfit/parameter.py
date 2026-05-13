"""Parameter — the atomic unit of every differentiable inverse problem.

Every input that *could* be refined is a :class:`Parameter`.  The forward
model never sees Python floats; it sees tensors that came out of a packed
parameter vector.  Refined components carry autograd; fixed components don't.

Promoted from ``midas_calibrate_v2.parameters.parameter``: the abstraction is
not specific to powder calibration — it's the shared substrate for joint
HEDM+powder calibration and any other torch-LM-based MIDAS inverse problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from midas_peakfit.transforms import Identity, Logit, Log, Transform


# ============================================================ Priors

class Prior:
    """Abstract prior. log_prob(x) returns the log-density at bounded x."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class GaussianPrior(Prior):
    mean: float
    std: float

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        return (-0.5 * z * z - torch.log(torch.tensor(self.std, dtype=x.dtype, device=x.device))
                - 0.5 * float(torch.log(torch.tensor(2 * 3.141592653589793))))


@dataclass
class HalfCauchyPrior(Prior):
    """For non-negative scale parameters (e.g., per-spot σ)."""

    scale: float

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # 2 / (πs) · 1/(1 + (x/s)²) for x > 0
        s = self.scale
        return torch.log(torch.tensor(2.0 / (3.141592653589793 * s),
                                      dtype=x.dtype, device=x.device)) \
            - torch.log(1.0 + (x.clamp(min=0.0) / s) ** 2)


@dataclass
class UniformPrior(Prior):
    lo: float
    hi: float

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        in_range = (x >= self.lo) & (x <= self.hi)
        ld = -torch.log(torch.tensor(self.hi - self.lo, dtype=x.dtype, device=x.device))
        return torch.where(in_range, ld, torch.full_like(x, -1e30))


# ============================================================ Parameter

@dataclass
class Parameter:
    """A scalar or tensor input to the forward model.

    Attributes
    ----------
    name
        Human-readable name (used for indexing and logging).
    init
        Initial value as a Python float, list, or torch.Tensor.  For non-scalar
        parameters (e.g., panel shifts), provide a list/tensor.
    refined
        If False, the value is held fixed; gradient does not flow.
    prior
        Optional :class:`Prior`.  For Bayesian inference, this contributes to
        the log-posterior.  For point estimation, it acts as a Gaussian /
        regularization term if used.
    bounds
        Optional ``(lo, hi)``.  When set, the default transform becomes
        :class:`Logit` over the box; otherwise :class:`Identity`.
    transform
        Override the default transform.
    shape
        Resolved shape (computed from ``init``).
    """

    name: str
    init: object   # float | list | torch.Tensor
    refined: bool = True
    prior: Optional[Prior] = None
    bounds: Optional[Tuple[float, float]] = None
    transform: Optional[Transform] = None

    shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.init, torch.Tensor):
            self.shape = tuple(self.init.shape)
        elif isinstance(self.init, (list, tuple)):
            arr = torch.as_tensor(self.init, dtype=torch.float64)
            self.init = arr
            self.shape = tuple(arr.shape)
        else:
            self.init = float(self.init)
            self.shape = ()
        if self.transform is None:
            if self.bounds is not None:
                lo, hi = self.bounds
                self.transform = Logit(lo, hi)
            else:
                self.transform = Identity()

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def init_tensor(self, dtype=torch.float64, device="cpu") -> torch.Tensor:
        if isinstance(self.init, torch.Tensor):
            return self.init.to(dtype=dtype, device=device).clone()
        return torch.full((), float(self.init), dtype=dtype, device=device)

    def make_logit_bounds(self, fallback_span: float = 1.0) -> Tuple[float, float]:
        """If user supplied no bounds, fabricate symmetric bounds around init."""
        if self.bounds is not None:
            return self.bounds
        if isinstance(self.init, torch.Tensor):
            v = float(self.init.flatten()[0])
        else:
            v = float(self.init)
        return (v - fallback_span, v + fallback_span)

    def __repr__(self) -> str:
        flag = "refined" if self.refined else "fixed"
        return f"Parameter({self.name!r}, init={self.init!r}, {flag}, shape={self.shape})"


__all__ = ["Parameter", "Prior", "GaussianPrior", "HalfCauchyPrior", "UniformPrior"]

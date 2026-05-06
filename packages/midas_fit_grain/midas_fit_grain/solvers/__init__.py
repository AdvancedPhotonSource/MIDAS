"""Solver registry.

Each entry takes the same call signature::

    minimize(closure, params, *, max_iter, ftol, xtol, **opts) -> SolveResult

``closure`` returns a scalar tensor (sum-of-squares of residuals). For L-M
the closure may also return raw residuals via the ``residuals`` kwarg; the
default registry assumes scalar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from .adam import minimize_adam
from .lbfgs import minimize_lbfgs
from .lm import minimize_lm
from .nelder_mead import minimize_nelder_mead


@dataclass
class SolveResult:
    final_loss: float
    n_iter: int
    converged: bool
    history: list[float]


# Each solver expects a different closure flavor:
#   "scalar_with_backward"  : closure returns a scalar tensor and ALREADY
#                             called .backward() on it. (LBFGS contract.)
#   "scalar_no_backward"    : closure returns a scalar tensor; the solver
#                             handles backward itself (or doesn't need it).
#                             ADAM and Nelder-Mead.
#   "residual_no_backward"  : closure returns the un-summed residual
#                             vector; LM does its own Jacobian.
_CLOSURE_KIND = {
    "lbfgs":       "scalar_with_backward",
    "adam":        "scalar_with_backward",   # ADAM does need autograd; the
                                              # closure_with_backward variant
                                              # keeps the graph live for it.
    "nelder_mead": "scalar_no_backward",
    "lm":          "residual_no_backward",
}

_REGISTRY = {
    "lbfgs":       minimize_lbfgs,
    "adam":        minimize_adam,
    "lm":          minimize_lm,
    "nelder_mead": minimize_nelder_mead,
}


def get_solver(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown solver '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def closure_kind(name: str) -> str:
    if name not in _CLOSURE_KIND:
        raise KeyError(f"Unknown solver '{name}'")
    return _CLOSURE_KIND[name]


def register(name: str, fn: Callable, *, closure_kind: str) -> None:
    _REGISTRY[name] = fn
    _CLOSURE_KIND[name] = closure_kind


__all__ = [
    "SolveResult", "get_solver", "closure_kind", "register",
    "minimize_lbfgs", "minimize_adam", "minimize_lm", "minimize_nelder_mead",
]

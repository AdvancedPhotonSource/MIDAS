"""Nelder–Mead simplex solver via SciPy.

This solver exists for **C parity**: it mirrors what
``FitPosOrStrainsOMP.c`` does (NLopt LN_NELDERMEAD). Don't pick it for
production runs — L-BFGS or L-M are faster — but it's invaluable when
diffing Python output against the C reference.

Operates purely on detached numpy values; no autograd.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
from scipy.optimize import minimize


def minimize_nelder_mead(
    closure: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    *,
    max_iter: int = 5000,
    ftol: float = 1e-5,
    xtol: float = 1e-5,
    **_,
):
    """Minimize ``loss(closure)`` via SciPy ``minimize(method='Nelder-Mead')``.

    The closure must compute the loss at the *current* values of
    ``params`` and return a scalar tensor. Backward is **not** required;
    Nelder–Mead is derivative-free.
    """
    if not params:
        raise ValueError("Nelder-Mead needs at least one parameter")

    sizes = [p.numel() for p in params]
    shapes = [p.shape for p in params]

    def _read_flat() -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in params])

    def _write_flat(flat: np.ndarray) -> None:
        i = 0
        for p, s, sh in zip(params, sizes, shapes):
            slab = flat[i:i + s].reshape(sh)
            p.detach().copy_(torch.from_numpy(slab).to(dtype=p.dtype, device=p.device))
            i += s

    saved = _read_flat()

    history: list[float] = []
    iters = 0

    def _f(flat_np: np.ndarray) -> float:
        nonlocal iters
        _write_flat(flat_np)
        with torch.no_grad():
            loss = closure()
        v = float(loss.detach().cpu().item())
        history.append(v)
        iters += 1
        return v

    res = minimize(
        _f, saved, method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "fatol": ftol,
            "xatol": xtol,
            "adaptive": True,
            "disp": False,
        },
    )

    _write_flat(res.x)

    return {
        "final_loss": float(res.fun),
        "n_iter": int(getattr(res, "nit", iters)),
        "converged": bool(res.success),
        "history": history,
    }

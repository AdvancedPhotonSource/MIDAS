"""ADAM solver wrapper.

ADAM is the right tool when the objective is noisy or ragged (e.g. between
re-association steps). Defaults are conservative; the refiner driver tunes
``lr`` per parameter group (position vs. orientation vs. lattice).
"""

from __future__ import annotations

from typing import Callable, List

import torch


def minimize_adam(
    closure: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    *,
    max_iter: int = 500,
    ftol: float = 1e-6,
    xtol: float = 1e-7,
    lr: float = 0.1,
    betas: tuple = (0.9, 0.999),
    **_,
):
    if not params:
        raise ValueError("ADAM needs at least one parameter")

    optimizer = torch.optim.Adam(params, lr=lr, betas=betas)

    history: list[float] = []
    prev_loss = float("inf")
    converged = False
    n_iter = 0
    last_x = [p.detach().clone() for p in params]
    n_below = 0

    for step in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        loss = closure()           # closure already called .backward()
        if not torch.isfinite(loss):
            break
        optimizer.step()

        loss_v = float(loss.detach())
        history.append(loss_v)
        n_iter = step + 1

        rel = abs(loss_v - prev_loss) / max(abs(prev_loss), 1e-12)
        dx = max(
            (p.detach() - lx).abs().max().item()
            for p, lx in zip(params, last_x)
        )
        if rel < ftol and dx < xtol:
            n_below += 1
            if n_below >= 5:        # 5 consecutive ftol-tight steps
                converged = True
                break
        else:
            n_below = 0
        prev_loss = loss_v
        last_x = [p.detach().clone() for p in params]

    return {
        "final_loss": history[-1] if history else float("inf"),
        "n_iter": n_iter,
        "converged": converged,
        "history": history,
    }

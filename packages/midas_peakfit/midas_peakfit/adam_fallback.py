"""Adam fallback optimizer for regions where LM diverges or saturates.

Used as a recovery path: if LM hits ``return_code != 0`` for a region, we
re-run those regions through Adam for a fixed number of steps. The result
is then taken as the final fit (with ``return_code = -1`` to flag that
the fit was non-converged-with-LM).
"""
from __future__ import annotations

import torch

from midas_peakfit.model import cost
from midas_peakfit.reparam import u_to_x, x_to_u


def adam_polish(
    x_init: torch.Tensor,  # [B, N]
    lo: torch.Tensor,
    hi: torch.Tensor,
    z: torch.Tensor,
    Rs: torch.Tensor,
    Etas: torch.Tensor,
    pixel_mask: torch.Tensor,
    n_peaks: int,
    *,
    n_steps: int = 50,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a few Adam steps in unconstrained space. Returns (x_final, cost_final)."""
    u = x_to_u(x_init, lo, hi).clone().detach().requires_grad_(True)
    optim = torch.optim.Adam([u], lr=lr)

    for _ in range(n_steps):
        optim.zero_grad(set_to_none=True)
        x = u_to_x(u, lo, hi)
        c = cost(x, z, Rs, Etas, pixel_mask, n_peaks).sum()
        c.backward()
        optim.step()

    with torch.no_grad():
        x_final = u_to_x(u, lo, hi)
        c_final = cost(x_final, z, Rs, Etas, pixel_mask, n_peaks)
    return x_final.detach(), c_final.detach()


__all__ = ["adam_polish"]

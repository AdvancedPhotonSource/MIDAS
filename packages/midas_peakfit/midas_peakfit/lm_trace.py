"""Per-iteration cost tracing for the LM solver, used for paper Fig 2.

The LM trajectory is interesting on three axes: (i) how cost decays with
iteration, (ii) how the decay speed scales with n_peaks, and (iii) how it
compares to gradient-only methods like Adam at the same n_peaks. We
record cost-vs-iteration during a normal LM run, gated behind an opt-in
flag so production runs are unaffected.

Usage::

    from midas_peakfit.lm_trace import LMTracer
    tracer = LMTracer()
    cfg = LMConfig(...)
    cfg._tracer = tracer  # attached as a private attribute; lm.py picks it up
    lm_solve(..., config=cfg)
    tracer.save("lm_trace.npz")

For Adam comparison see :func:`adam_trace_solve` below; it runs the same
problem with Adam and records the same cost-vs-iter trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class LMTracer:
    """Logs per-iteration cost for every region in the bucket.

    ``costs[(n_peaks, M, B)] = (n_iter+1, B)`` ndarray (initial cost in
    row 0, final-iter cost in last row). ``n_active[(...,)]`` records
    how many regions were still active at the start of each iteration.
    """

    costs: Dict[tuple, np.ndarray] = field(default_factory=dict)
    n_active: Dict[tuple, np.ndarray] = field(default_factory=dict)

    def begin(self, key: tuple, B: int, n_iter_max: int) -> None:
        self.costs[key] = np.full((n_iter_max + 1, B), np.nan, dtype=np.float64)
        self.n_active[key] = np.zeros(n_iter_max + 1, dtype=np.int64)

    def log(self, key: tuple, it: int, cost: torch.Tensor, n_active: int) -> None:
        if key not in self.costs:
            return
        c = cost.detach().to(torch.float64).cpu().numpy()
        self.costs[key][it, :len(c)] = c
        self.n_active[key][it] = n_active

    def save(self, path: str) -> None:
        out: Dict[str, np.ndarray] = {}
        for k, v in self.costs.items():
            label = f"costs_n{k[0]}_M{k[1]}_B{k[2]}"
            out[label] = v
            out[f"nactive_n{k[0]}_M{k[1]}_B{k[2]}"] = self.n_active[k]
        np.savez(path, **out)


def adam_trace_solve(
    x_init: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    z: torch.Tensor,
    Rs: torch.Tensor,
    Etas: torch.Tensor,
    pixel_mask: torch.Tensor,
    n_peaks: int,
    *,
    n_steps: int = 200,
    lr: float = 1e-2,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run plain Adam in the unbounded ``u`` space and log cost per step.

    Returns ``(x_final, cost_history)`` where ``cost_history`` is
    shape ``(n_steps + 1, B)``. Used to compare LM's quadratic-local
    convergence against gradient descent at the same starting point.
    """
    from midas_peakfit.model import residuals
    from midas_peakfit.reparam import u_to_x, x_to_u

    u = x_to_u(x_init, lo, hi).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=lr)

    history = np.full((n_steps + 1, x_init.shape[0]), np.nan, dtype=np.float64)
    with torch.no_grad():
        x = u_to_x(u, lo, hi)
        r = residuals(x, z, Rs, Etas, pixel_mask, n_peaks)
        history[0, :] = (r * r).sum(dim=-1).detach().to(torch.float64).cpu().numpy()

    for it in range(n_steps):
        optimizer.zero_grad()
        x = u_to_x(u, lo, hi)
        r = residuals(x, z, Rs, Etas, pixel_mask, n_peaks)
        cost = (r * r).sum()
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            r_after = residuals(u_to_x(u, lo, hi), z, Rs, Etas, pixel_mask, n_peaks)
            c_after = (r_after * r_after).sum(dim=-1).detach().to(torch.float64).cpu().numpy()
            history[it + 1, :] = c_after

    return u_to_x(u, lo, hi).detach(), history


__all__ = ["LMTracer", "adam_trace_solve"]

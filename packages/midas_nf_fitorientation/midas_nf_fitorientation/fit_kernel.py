"""Shared L-BFGS driver used by all three fit entry points.

Provides one function, :func:`run_lbfgs`, that takes a closure and a
list of leaf parameters and runs PyTorch's L-BFGS optimiser with
strong-Wolfe line search to convergence. Lifts a few common patterns
that would otherwise be duplicated across the three drivers:

- Closure must clear .grad on every leaf, build the loss, call
  ``.backward()``, and return the loss tensor.
- Optionally early-exit when loss falls below ``stop_loss``.
- Returns the final loss + the number of L-BFGS steps actually taken.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import torch


@dataclass
class LBFGSConfig:
    """Per-call L-BFGS settings.

    Attributes
    ----------
    lr : float
        Step size for the line search. ``1.0`` works well with
        strong-Wolfe; smaller values (0.3–0.5) are safer for joint
        Euler+calibration phases.
    max_iter : int
        Inner iterations per ``optimizer.step()`` call.
    max_outer : int
        Outer step calls. Each outer step internally does up to
        ``max_iter`` line-search iterations.
    tolerance_grad : float
        L-BFGS gradient termination threshold.
    tolerance_change : float
        L-BFGS function-value change threshold.
    stop_loss : float
        Early-exit if the loss falls below this. ``1e-4`` ≈ overlap
        ≥ 0.9999.
    history_size : int
        L-BFGS memory length.
    """
    lr: float = 1.0
    max_iter: int = 20
    max_outer: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    stop_loss: float = 1e-4
    history_size: int = 20


@dataclass
class LBFGSResult:
    final_loss: float
    n_outer_steps: int
    converged: bool


def run_lbfgs(
    closure: Callable[[], torch.Tensor],
    leaves: Iterable[torch.Tensor],
    config: LBFGSConfig | None = None,
) -> LBFGSResult:
    """Run L-BFGS until convergence or ``max_outer`` is exhausted.

    The ``closure`` must:
    1. zero ``.grad`` on every leaf,
    2. compute and return the loss tensor,
    3. call ``loss.backward()`` before returning.

    Returns the last loss reached and the number of outer steps taken.
    """
    cfg = config or LBFGSConfig()
    leaves_list: List[torch.Tensor] = list(leaves)
    optimizer = torch.optim.LBFGS(
        leaves_list,
        lr=cfg.lr,
        max_iter=cfg.max_iter,
        history_size=cfg.history_size,
        tolerance_grad=cfg.tolerance_grad,
        tolerance_change=cfg.tolerance_change,
        line_search_fn="strong_wolfe",
    )

    last_loss = float("inf")
    converged = False
    for step in range(cfg.max_outer):
        loss = optimizer.step(closure)
        last_loss = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        if last_loss < cfg.stop_loss:
            converged = True
            return LBFGSResult(
                final_loss=last_loss,
                n_outer_steps=step + 1,
                converged=True,
            )

    return LBFGSResult(
        final_loss=last_loss,
        n_outer_steps=cfg.max_outer,
        converged=converged,
    )

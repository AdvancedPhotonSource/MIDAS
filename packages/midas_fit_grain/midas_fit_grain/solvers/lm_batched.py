"""Batched, GPU-resident Levenberg-Marquardt solver.

Optimises ``B`` independent grains in parallel — each grain has its own
9-DOF problem (3 pos + 3 euler + 3 lattice; +3 if the lattice has 6 DOFs
unlocked, but we treat the standard 9-param case here). All work happens
on the GPU; one ``.cpu()`` sync per outer iter (for convergence check).

This is the structural fix for park22-scale refinement where
``torch.optim.LBFGS`` was bottlenecked at ~290 s by per-line-search-probe
Python-coordination overhead. Batched LM cuts that loop entirely:

  per outer iter:
    1. one batched forward to get residuals    (B, n_res)
    2. P=12 batched forwards for finite-diff Jacobian
    3. one batched solve                         (B, P)
    4. accept/reject per grain via torch.where
    5. one sync at convergence-check (every K iters)

vs LBFGS:
  per outer iter:
    1. ~5 forward + backward for line-search probes
    2. ~5 .item() syncs for line-search comparisons
    3. CPU-side Wolfe condition state machine

Per-grain numerical Jacobian (P+1=13 forwards per outer iter) is acceptable
because each forward is already batched across all B grains — the marginal
cost is just one extra batched kernel launch chain per param dim.
"""
from __future__ import annotations

from typing import Callable, List

import torch


def minimize_lm_batched(
    residual_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    pos_scaled: torch.Tensor,        # (B, 3)  position / pos_scale
    euler: torch.Tensor,             # (B, 3)
    lattice: torch.Tensor,           # (B, 6)
    *,
    pos_scale: float = 100.0,
    max_iter: int = 50,
    ftol: float = 1e-5,
    xtol: float = 1e-7,
    lambda_init: float = 1e-3,
    lambda_up: float = 10.0,
    lambda_down: float = 0.5,
    lambda_max: float = 1e10,
    lambda_min: float = 1e-15,
    converge_check_every: int = 5,
    eps_rel: float = 1e-6,
    eps_abs: float = 1e-9,
    active_mask: torch.Tensor | None = None,   # (P,) bool — which of the 12 params are active
):
    """Batched LM. Returns updated (pos_scaled, euler, lattice) in-place.

    ``residual_fn(pos, euler, lattice)`` must return a ``(B, n_res)`` tensor
    of un-summed residuals. ``B`` matches the grain count.

    The 12 free params per grain are flattened as
    ``[px, py, pz, e1, e2, e3, a, b, c, alpha, beta, gamma]``.
    ``active_mask`` lets the caller restrict the optimization to a subset
    (e.g., position-only phase, lattice-only phase) without having to
    re-pack tensors. Inactive params are held fixed; their Jacobian
    columns are skipped, and the per-grain solve is over the active subset
    only.

    Convergence is per-grain: a grain is "done" when both
      * |Δr|² / |r|² < ftol  (relative loss change)
      * ‖δ‖∞ / ‖p‖∞ < xtol   (relative param change)
    AND the most recent step was accepted. Inactive grains keep iterating
    via where() — no early-exit per grain (would break batching), just
    a global outer break when *all* grains have converged.
    """
    B = pos_scaled.shape[0]
    device = pos_scaled.device
    dtype = pos_scaled.dtype
    P = 12

    if active_mask is None:
        active_mask = torch.ones(P, dtype=torch.bool, device=device)
    n_active = int(active_mask.sum().item())
    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)   # (n_active,)

    # Pack into one (B, P) flat tensor.
    flat = torch.cat([pos_scaled, euler, lattice], dim=1).contiguous()
    if flat.shape[1] != P:
        raise ValueError(f"expected {P} packed params, got {flat.shape[1]}")

    def _eval(flat_in: torch.Tensor) -> torch.Tensor:
        return residual_fn(
            flat_in[:, :3] * pos_scale,
            flat_in[:, 3:6],
            flat_in[:, 6:],
        )

    with torch.no_grad():
        r_cur = _eval(flat)                                # (B, n_res)
        n_res = r_cur.shape[1]
        cost_cur = (r_cur * r_cur).sum(dim=1)              # (B,)
        prev_cost = cost_cur.clone()
        lam = torch.full((B,), lambda_init, dtype=dtype, device=device)
        converged = torch.zeros(B, dtype=torch.bool, device=device)

        n_iter = 0
        all_done = False

        for outer in range(max_iter):
            n_iter = outer + 1

            # Numerical Jacobian: only over active param columns.
            # Step size per param scales with magnitude (NumPy-LSQ style).
            J = torch.zeros(B, n_res, n_active, dtype=dtype, device=device)
            for j, p_idx in enumerate(active_idx.tolist()):
                eps_col = (flat[:, p_idx].abs() + 1.0) * eps_rel + eps_abs
                flat_plus = flat.clone()
                flat_plus[:, p_idx] = flat_plus[:, p_idx] + eps_col
                r_plus = _eval(flat_plus)
                J[:, :, j] = (r_plus - r_cur) / eps_col.unsqueeze(1)

            # Batched LM normal equations on the active subset.
            JtJ = torch.matmul(J.transpose(1, 2), J)         # (B, n_a, n_a)
            Jtr = torch.matmul(J.transpose(1, 2),
                               r_cur.unsqueeze(-1)).squeeze(-1)  # (B, n_a)

            # Marquardt damping: λ * diag(JᵀJ).
            diag_jtj = JtJ.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
            damping = lam.unsqueeze(-1) * diag_jtj           # (B, n_a)
            A = JtJ + torch.diag_embed(damping)              # (B, n_a, n_a)

            # Batched solve. ``torch.linalg.solve`` is batched over the
            # leading dim; failure on any grain → fallback to lstsq.
            try:
                delta_active = -torch.linalg.solve(A, Jtr.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                delta_active = -torch.linalg.lstsq(A, Jtr.unsqueeze(-1)).solution.squeeze(-1)

            # Scatter active deltas into a (B, P) full delta.
            delta = torch.zeros_like(flat)
            delta[:, active_idx] = delta_active

            flat_trial = flat + delta
            r_trial = _eval(flat_trial)
            cost_trial = (r_trial * r_trial).sum(dim=1)      # (B,)

            # Per-grain accept/reject.
            accept = cost_trial < cost_cur                    # (B,)
            accept_e = accept.unsqueeze(-1)                   # (B, 1)

            flat = torch.where(accept_e, flat_trial, flat)
            r_cur = torch.where(accept_e, r_trial, r_cur)
            prev_cost = cost_cur
            cost_cur = torch.where(accept, cost_trial, cost_cur)
            lam = torch.where(accept,
                              (lam * lambda_down).clamp_min(lambda_min),
                              (lam * lambda_up).clamp_max(lambda_max))

            # Convergence check (per-grain, AND-combine across criteria).
            # Only sync to CPU every K outer iters to keep the GPU busy.
            if (outer + 1) % converge_check_every == 0:
                with torch.no_grad():
                    rel_loss = (prev_cost - cost_cur).abs() / prev_cost.clamp_min(1e-30)
                    rel_x = delta.abs().amax(dim=1) / (
                        flat.abs().amax(dim=1).clamp_min(1e-30)
                    )
                    new_done = accept & (rel_loss < ftol) & (rel_x < xtol)
                    converged = converged | new_done
                    # ONE sync: did all grains converge?
                    all_done = bool(converged.all().item())
                if all_done:
                    break

        new_pos_scaled = flat[:, :3].contiguous()
        new_euler = flat[:, 3:6].contiguous()
        new_lattice = flat[:, 6:].contiguous()

    return {
        "pos_scaled": new_pos_scaled,
        "euler": new_euler,
        "lattice": new_lattice,
        "n_iter": n_iter,
        "converged": all_done,
        "final_cost": cost_cur,
    }

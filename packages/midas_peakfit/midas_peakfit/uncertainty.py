"""Per-parameter uncertainty from the inverse Hessian at LM convergence.

For least-squares with Gaussian residuals, the parameter covariance is

    Cov_u = σ_r² × (J^T J)^{-1}

where J = ∂r/∂u is the residual Jacobian at the converged point and
σ_r² = (r·r) / max(M_eff − N, 1) is the residual-variance estimator
(M_eff is the count of unmasked pixels per region).

We optimize in unbounded ``u`` space (sigmoid reparameterization) but report
σ in the bounded ``x`` space the user thinks in. Because the reparam is
element-wise

    x_i = lo_i + (hi_i − lo_i) × σ(u_i)

the Jacobian D = ∂x/∂u is diagonal, and

    Cov_x = D × Cov_u × D     →     σ_x[i] = |D_ii| × √diag(Cov_u)[i]

When a parameter is pinned to a bound, σ(u) saturates and D_ii → 0; we
report σ_x = 0 there (the model has no informational degrees of freedom
left in that direction). When (J^T J) is rank-deficient (degenerate peaks,
zero curvature direction) Cholesky fails and we report σ_x = NaN.

This is a frequentist 1-σ standard error under Gaussian-residual /
Cramér–Rao assumptions. Validity at low photon counts and on strongly
overlapping peaks should be checked empirically (see paper §5.1).
"""
from __future__ import annotations

from typing import Tuple

import torch


def compute_param_sigma(
    u: torch.Tensor,            # [B, N]   — converged params in u-space
    lo: torch.Tensor,           # [B, N]
    hi: torch.Tensor,           # [B, N]
    J: torch.Tensor,            # [B, M, N] — ∂r/∂u at u
    r: torch.Tensor,            # [B, M]    — residual at u
    pixel_mask: torch.Tensor,   # [B, M]    — 1.0 active, 0.0 padded
    *,
    ridge: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sigma_x, dof) where

      sigma_x  : [B, N] — 1-σ in the bounded x-space (NaN where rank-deficient)
      dof      : [B]    — degrees of freedom (M_eff − N) per region

    Numerical strategy: Cholesky-factor (J^T J + ridge·max_diag·I) and
    invert via two triangular solves to get diag((J^T J)^{-1}).
    """
    B, N = u.shape
    M = r.shape[-1]

    # Active pixel count per region; pixel_mask carries 0/1 with float dtype.
    n_active = pixel_mask.sum(dim=-1)            # [B]
    dof = (n_active - float(N)).clamp(min=1.0)   # [B]

    # σ_r² per region (residual variance estimator).
    rss = (r * r).sum(dim=-1)                    # [B]
    sigma_r2 = rss / dof                          # [B]

    # J^T J. Match the precision policy used in lm.py: assemble in fp32 with
    # TF32 matmul on CUDA, cast to fp64 for Cholesky stability. On CPU or
    # when J is fp32 we just use the native dtype.
    if J.device.type == "cuda" and J.dtype == torch.float64:
        J32 = J.float()
        Jt32 = J32.transpose(-1, -2)
        H = (Jt32 @ J32).double()
    else:
        Jt = J.transpose(-1, -2)
        H = Jt @ J

    # Tiny diagonal ridge so a numerically rank-deficient Hessian still
    # gives a finite Cholesky. The ridge is scaled by max(diag(H)) so it's
    # invariant to overall problem scale.
    diagH = torch.diagonal(H, dim1=-2, dim2=-1)   # [B, N]
    max_diag = diagH.amax(dim=-1).clamp(min=1.0)  # [B]
    eye = torch.eye(N, dtype=H.dtype, device=H.device)
    H_reg = H + (ridge * max_diag).view(-1, 1, 1) * eye

    L_chol, info = torch.linalg.cholesky_ex(H_reg)        # [B, N, N], [B]
    # cholesky_solve(I, L) gives H_reg^{-1}. The diagonal is what we need.
    inv_H = torch.cholesky_solve(eye.expand(B, -1, -1), L_chol)  # [B, N, N]
    diag_invH = torch.diagonal(inv_H, dim1=-2, dim2=-1)          # [B, N]

    # Cov_u = σ_r² × (J^T J)^{-1}; we only need its diagonal.
    var_u = sigma_r2.unsqueeze(-1) * diag_invH               # [B, N]
    # Numerical noise can drop var_u slightly negative for tightly-converged
    # near-singular directions; clamp.
    var_u = var_u.clamp(min=0.0)
    sigma_u = var_u.sqrt()

    # Map to x-space: |dx/du| = (hi − lo) × σ(u) × (1 − σ(u)).
    s = torch.sigmoid(u)
    dxdu = (hi - lo) * s * (1.0 - s)             # [B, N]
    sigma_x = dxdu.abs() * sigma_u

    # Mark singular regions as NaN.
    bad = info != 0
    if bad.any():
        sigma_x[bad] = float("nan")

    return sigma_x, dof


__all__ = ["compute_param_sigma"]

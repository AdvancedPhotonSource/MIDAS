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

This module also exposes a complementary, **model-free** sensitivity from
the first three angular moments — the shot-noise-limited σ derived in
Modregger et al., J. Appl. Cryst. 58, 1653 (2025). The two estimators are
independent: the Hessian σ is the Cramér–Rao bound under the
Pseudo-Voigt model; the moment σ assumes only Poisson statistics on the
pixel counts. They should agree to within the constant ~1.5× factor
documented in the paper for any well-behaved single peak.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


# Photon-count regime thresholds per Modregger et al. 2025 (§3 + Appendix A
# of the midas-peakfit paper). M_0 ≳ 25 is the regime in which both
# moment-based and Hessian-based σ are well-calibrated; below ~5 photons
# per peak both estimators become systematically optimistic and a
# likelihood-based treatment is required.
QUALITY_OK = 0           # M_0 ≥ M0_MARGINAL_THRESHOLD
QUALITY_MARGINAL = 1     # M0_DEEP_POISSON_THRESHOLD ≤ M_0 < M0_MARGINAL_THRESHOLD
QUALITY_DEEP_POISSON = 2  # M_0 < M0_DEEP_POISSON_THRESHOLD

M0_DEEP_POISSON_THRESHOLD = 5.0
M0_MARGINAL_THRESHOLD = 25.0


def compute_moment_sigma(
    M0: Union[np.ndarray, float],
    M2: Union[np.ndarray, float],
    M4: Union[np.ndarray, float],
    dx: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shot-noise-limited 1-σ uncertainty for the first three angular moments.

    Closed-form result from Modregger et al., J. Appl. Cryst. 58, 1653 (2025),
    equations (6)–(8). Applies to a 1-D photon-count distribution sampled at
    spacing ``dx``:

        u(M_0) = sqrt(dx · M_0)
        u(M_1) = sqrt(dx · M_2 / M_0)
        u(M_2) = sqrt(dx · (M_4 − M_2^2) / M_0)

    These are the photon-shot-noise floor under Poisson statistics — the
    smallest uncertainty achievable on a single diffraction frame. Valid
    when M_0 ≳ 25; for lower counts use a Poisson-likelihood treatment.

    Parameters
    ----------
    M0, M2, M4 : array-like or float
        Zeroth, second, and fourth moments of the (background-subtracted)
        per-peak intensity distribution along the axis of interest. Shapes
        must broadcast; M_0 is in photon counts, M_2 in ``dx``-units²,
        M_4 in ``dx``-units⁴.
    dx : float, default 1.0
        Sampling spacing (e.g. detector pixel size in pixels, or angular
        pixel width in radians/degrees). The returned uncertainties carry
        the same units as M_1 / sqrt(M_2) along that axis.

    Returns
    -------
    u_M0, u_M1, u_M2 : ndarrays (or floats)
        Shot-noise 1-σ uncertainties on M_0, M_1, M_2. ``np.nan`` where
        M_0 ≤ 0 (no signal — formulas not applicable).

    Notes
    -----
    The argument of the square root in u(M_2), namely ``M_4 − M_2^2``, is
    non-negative by the Cauchy–Schwarz inequality (equivalent to
    kurtosis ≥ 1, see Modregger 2025 §2). We clip tiny negative values
    that arise from floating-point cancellation.
    """
    M0_a = np.asarray(M0, dtype=np.float64)
    M2_a = np.asarray(M2, dtype=np.float64)
    M4_a = np.asarray(M4, dtype=np.float64)

    valid = M0_a > 0.0
    u_M0 = np.where(valid, np.sqrt(dx * np.maximum(M0_a, 0.0)), np.nan)
    # Guard division by zero where M_0 ≤ 0.
    safe_M0 = np.where(valid, M0_a, 1.0)
    u_M1 = np.where(valid, np.sqrt(dx * M2_a / safe_M0), np.nan)
    # M_4 − M_2^2 ≥ 0 in exact arithmetic; clip tiny negative roundoff.
    m4_m2sq = np.maximum(M4_a - M2_a * M2_a, 0.0)
    u_M2 = np.where(valid, np.sqrt(dx * m4_m2sq / safe_M0), np.nan)

    if M0_a.ndim == 0:
        return float(u_M0), float(u_M1), float(u_M2)
    return u_M0, u_M1, u_M2


def classify_peak_quality(
    M0: Union[np.ndarray, float],
) -> Union[np.ndarray, int]:
    """Classify per-peak photon-count regime per Modregger 2025.

    Returns an ``int8`` array (or scalar) with values

        0 = QUALITY_OK            (M_0 ≥ 25 photons — both Hessian and
                                   moment σ are well-calibrated)
        1 = QUALITY_MARGINAL      (5 ≤ M_0 < 25 — σ may be optimistic
                                   by a small constant factor)
        2 = QUALITY_DEEP_POISSON  (M_0 < 5 — Gaussian-residual / inverse-
                                   Hessian assumption breaks down;
                                   downstream code should down-weight
                                   or drop these peaks)

    The thresholds match the ranges discussed in the paper's Appendix A
    "Practical caveats" section.
    """
    M0_a = np.asarray(M0, dtype=np.float64)
    flag = np.where(
        M0_a < M0_DEEP_POISSON_THRESHOLD, QUALITY_DEEP_POISSON,
        np.where(M0_a < M0_MARGINAL_THRESHOLD, QUALITY_MARGINAL, QUALITY_OK),
    ).astype(np.int8)
    if M0_a.ndim == 0:
        return int(flag)
    return flag


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


__all__ = [
    "compute_param_sigma",
    "compute_moment_sigma",
    "classify_peak_quality",
    "QUALITY_OK",
    "QUALITY_MARGINAL",
    "QUALITY_DEEP_POISSON",
    "M0_DEEP_POISSON_THRESHOLD",
    "M0_MARGINAL_THRESHOLD",
]

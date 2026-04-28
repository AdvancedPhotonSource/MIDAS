"""Analytical Jacobian of the Pseudo-Voigt residual.

Replaces ``torch.func.jacrev`` (autograd reverse-mode) with closed-form
derivatives. For batched LM with B regions × M pixels × N=1+8P parameters,
the autograd path materialises full intermediate tensors per backward pass;
the analytical version computes everything in a single forward sweep using
shared intermediates (``G``, ``L``, ``A``, ``B``, ``dR``, ``dE``).

The reparameterization (``x = lo + (hi-lo) × σ(u)``) is incorporated by
multiplying each column of ``J_x`` by ``dx_k/du_k = (hi-lo) × t × (1-t)``,
where ``t = σ(u)``.
"""
from __future__ import annotations

import torch


def residuals_and_jacobian_u(
    u: torch.Tensor,         # [B, N]
    lo: torch.Tensor,        # [B, N]
    hi: torch.Tensor,        # [B, N]
    z: torch.Tensor,         # [B, M]
    Rs: torch.Tensor,        # [B, M]
    Etas: torch.Tensor,      # [B, M]
    pixel_mask: torch.Tensor,# [B, M]  (0 or 1)
    n_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residuals and ∂r/∂u in one fused forward pass.

    Returns:
        r:  [B, M]  — residual = (model − z) × pixel_mask
        J:  [B, M, N]  — ∂r_i/∂u_k

    The Jacobian is computed analytically; no autograd graph is built.
    All tensors must be on the same device and dtype.
    """
    B, N = u.shape
    M = z.shape[-1]
    P = n_peaks
    assert N == 1 + 8 * P

    # ── Reparameterization ────────────────────────────────────────
    span = hi - lo                            # [B, N]
    t = torch.sigmoid(u)                      # [B, N]
    x = lo + span * t                         # [B, N]
    dx_du = span * t * (1.0 - t)              # [B, N]

    # Split parameters
    bg = x[..., 0]                            # [B]
    rest = x[..., 1:].reshape(B, P, 8)
    Imax = rest[..., 0]                       # [B, P]
    R = rest[..., 1]                          # [B, P]
    Eta = rest[..., 2]                        # [B, P]
    Mu = rest[..., 3]                         # [B, P]
    sgR = rest[..., 4]
    slR = rest[..., 5]
    sgE = rest[..., 6]
    slE = rest[..., 7]

    # Floor at a dtype-appropriate epsilon to avoid div-by-zero. For fp32
    # the LM bounds keep σ ≥ 0.005, so 1e-10 is a safe floor; for fp64
    # we keep the historical 1e-12.
    eps_val = 1e-12 if u.dtype == torch.float64 else 1e-10
    eps = torch.tensor(eps_val, dtype=u.dtype, device=u.device)
    sgR2 = sgR * sgR + eps                    # [B, P]
    slR2 = slR * slR + eps
    sgE2 = sgE * sgE + eps
    slE2 = slE * slE + eps

    # ── Per-pixel × per-peak shared terms ────────────────────────
    # Broadcast: pixel index M, peak index P
    # Rs[:, :, None]: [B, M, 1]; R[:, None, :]: [B, 1, P]
    dR = Rs.unsqueeze(-1) - R.unsqueeze(-2)   # [B, M, P]
    dE = Etas.unsqueeze(-1) - Eta.unsqueeze(-2)
    dR2 = dR * dR
    dE2 = dE * dE

    sgR2_b = sgR2.unsqueeze(-2)               # [B, 1, P]
    slR2_b = slR2.unsqueeze(-2)
    sgE2_b = sgE2.unsqueeze(-2)
    slE2_b = slE2.unsqueeze(-2)

    A = 1.0 + dR2 / slR2_b                    # [B, M, P]
    Bx = 1.0 + dE2 / slE2_b
    L = 1.0 / (A * Bx)                        # [B, M, P]
    G = torch.exp(-0.5 * (dR2 / sgR2_b + dE2 / sgE2_b))  # [B, M, P]

    Mu_b = Mu.unsqueeze(-2)                   # [B, 1, P]
    Imax_b = Imax.unsqueeze(-2)
    profile = Imax_b * (Mu_b * L + (1.0 - Mu_b) * G)  # [B, M, P]
    model = bg.unsqueeze(-1) + profile.sum(dim=-1)    # [B, M]
    r = (model - z) * pixel_mask              # [B, M]

    # ── Build J_x ∈ [B, M, N] in u-coordinates ───────────────────
    # We compute J_x first, then multiply by dx_du for chain rule.
    # Allocate output and fill column-by-column.
    Jx = torch.zeros((B, M, N), dtype=u.dtype, device=u.device)

    # ∂r/∂bg = 1 (× mask)
    Jx[:, :, 0] = pixel_mask

    # Per-peak partial derivatives.
    # dM/d(Imax_j) = (μ L + (1-μ) G)
    Jx_Imax = (Mu_b * L + (1.0 - Mu_b) * G)         # [B, M, P]

    # dM/d(R_j) = Imax × [μ × ∂L/∂R_j + (1-μ) × ∂G/∂R_j]
    # ∂G/∂R_j = G × dR / sgR²  (sign: dR = R_p − R_j, ∂dR/∂R_j = −1)
    dG_dRj = G * (dR / sgR2_b)
    # ∂L/∂R_j: L = 1/(A·B); ∂L/∂R_j = -∂A/∂R_j × L/A = (2·dR/slR²) × (L/A)
    dL_dRj = (2.0 * dR / slR2_b) * (L / A)
    Jx_Rj = Imax_b * (Mu_b * dL_dRj + (1.0 - Mu_b) * dG_dRj)

    # dM/d(η_j)
    dG_dEj = G * (dE / sgE2_b)
    dL_dEj = (2.0 * dE / slE2_b) * (L / Bx)
    Jx_Ej = Imax_b * (Mu_b * dL_dEj + (1.0 - Mu_b) * dG_dEj)

    # dM/d(μ_j) = Imax × (L − G)
    Jx_Mu = Imax_b * (L - G)

    # dM/d(σGR_j) = Imax × (1-μ) × ∂G/∂σGR
    # ∂G/∂σGR = G × dR² / σGR³ = G × dR² / (σGR × σGR²)
    # We have sgR² with eps; for derivative use sgR*sgR2_b = σGR × σGR² (within eps).
    sgR_b = sgR.unsqueeze(-2)
    dG_dsgR = G * (dR2 / (sgR_b * sgR2_b))
    Jx_sgR = Imax_b * (1.0 - Mu_b) * dG_dsgR

    # dM/d(σLR_j) = Imax × μ × ∂L/∂σLR
    # ∂L/∂σLR = (2·dR²/σLR³) × (L/A)
    slR_b = slR.unsqueeze(-2)
    dL_dslR = (2.0 * dR2 / (slR_b * slR2_b)) * (L / A)
    Jx_slR = Imax_b * Mu_b * dL_dslR

    # dM/d(σGEta_j)
    sgE_b = sgE.unsqueeze(-2)
    dG_dsgE = G * (dE2 / (sgE_b * sgE2_b))
    Jx_sgE = Imax_b * (1.0 - Mu_b) * dG_dsgE

    # dM/d(σLEta_j)
    slE_b = slE.unsqueeze(-2)
    dL_dslE = (2.0 * dE2 / (slE_b * slE2_b)) * (L / Bx)
    Jx_slE = Imax_b * Mu_b * dL_dslE

    # Pack per-peak Jacobian columns into Jx, slot order matching split_params:
    # 1+8j+0: Imax, +1: R, +2: η, +3: μ, +4: σGR, +5: σLR, +6: σGEta, +7: σLEta
    # Each Jx_* has shape [B, M, P]; reshape to [B, M, P, 1] and stack
    pmask_3 = pixel_mask.unsqueeze(-1)          # [B, M, 1]
    cols_per_peak = torch.stack(
        [
            Jx_Imax * pmask_3,
            Jx_Rj * pmask_3,
            Jx_Ej * pmask_3,
            Jx_Mu * pmask_3,
            Jx_sgR * pmask_3,
            Jx_slR * pmask_3,
            Jx_sgE * pmask_3,
            Jx_slE * pmask_3,
        ],
        dim=-1,
    )  # [B, M, P, 8]
    Jx[:, :, 1:] = cols_per_peak.reshape(B, M, P * 8)

    # Chain rule: J_u = J_x × dx_du (broadcast over M)
    # dx_du has shape [B, N]; expand to [B, 1, N]
    Ju = Jx * dx_du.unsqueeze(-2)
    return r, Ju


__all__ = ["residuals_and_jacobian_u"]

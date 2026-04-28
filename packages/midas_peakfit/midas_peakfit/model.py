"""Differentiable Pseudo-Voigt model — exact form from the C objective.

Replicates ``peakFittingObjectiveFunction`` in PeaksFittingOMPZarrRefactor.c
(lines 711-775). The Lorentzian is the *factored* form (NOT the standard
radial-symmetric 2D Lorentzian):

    G_j(r,η) = exp(-0.5 × [(r-R_j)²/σGR_j² + (η-η_j)²/σGEta_j²])
    L_j(r,η) = 1 / [(1 + (r-R_j)²/σLR_j²) × (1 + (η-η_j)²/σLEta_j²)]
    I(r,η)   = bg + Σ_j Imax_j × [μ_j × L_j + (1-μ_j) × G_j]

All inputs are batched-friendly tensors. Padded pixels (where
``pixel_mask == 0``) contribute zero residual.
"""
from __future__ import annotations

import torch

# Per-peak parameter offsets within the [B, 1 + 8P] flat tensor (matches C):
#   x[0]: bg
#   x[1 + 8j + 0]: Imax_j      (slot 0)
#   x[1 + 8j + 1]: R_j         (slot 1)
#   x[1 + 8j + 2]: Eta_j       (slot 2)
#   x[1 + 8j + 3]: Mu_j        (slot 3)
#   x[1 + 8j + 4]: SigmaGR_j   (slot 4)
#   x[1 + 8j + 5]: SigmaLR_j   (slot 5)
#   x[1 + 8j + 6]: SigmaGEta_j (slot 6)
#   x[1 + 8j + 7]: SigmaLEta_j (slot 7)


def split_params(x: torch.Tensor, n_peaks: int):
    """Split the 1 + 8P flat parameter vector into named per-peak tensors.

    ``x`` shape: ``[B, 1 + 8P]``.
    Returns: ``bg`` of shape ``[B]`` and per-peak tensors of shape ``[B, P]``.
    """
    bg = x[..., 0]
    rest = x[..., 1:].reshape(*x.shape[:-1], n_peaks, 8)
    Imax = rest[..., 0]
    R = rest[..., 1]
    Eta = rest[..., 2]
    Mu = rest[..., 3]
    sgR = rest[..., 4]
    slR = rest[..., 5]
    sgE = rest[..., 6]
    slE = rest[..., 7]
    return bg, Imax, R, Eta, Mu, sgR, slR, sgE, slE


def forward_pseudo_voigt(
    x: torch.Tensor,  # [B, 1 + 8P]
    Rs: torch.Tensor,  # [B, M]
    Etas: torch.Tensor,  # [B, M]
    n_peaks: int,
) -> torch.Tensor:
    """Compute model intensity per pixel for each region in the batch.

    Returns shape ``[B, M]``.
    """
    bg, Imax, R, Eta, Mu, sgR, slR, sgE, slE = split_params(x, n_peaks)
    # Broadcast: pixel index M, peak index P
    # Rs[:, :, None]: [B, M, 1]; R[:, None, :]: [B, 1, P]
    dR = Rs.unsqueeze(-1) - R.unsqueeze(-2)  # [B, M, P]
    dE = Etas.unsqueeze(-1) - Eta.unsqueeze(-2)
    dR2 = dR * dR
    dE2 = dE * dE

    eps_val = 1e-12 if x.dtype == torch.float64 else 1e-10
    eps = torch.tensor(eps_val, dtype=x.dtype, device=x.device)
    invSGR2 = 1.0 / (sgR * sgR + eps).unsqueeze(-2)
    invSLR2 = 1.0 / (slR * slR + eps).unsqueeze(-2)
    invSGE2 = 1.0 / (sgE * sgE + eps).unsqueeze(-2)
    invSLE2 = 1.0 / (slE * slE + eps).unsqueeze(-2)

    L = 1.0 / ((dR2 * invSLR2 + 1.0) * (dE2 * invSLE2 + 1.0))
    G = torch.exp(-0.5 * (dR2 * invSGR2 + dE2 * invSGE2))
    profile = Imax.unsqueeze(-2) * (Mu.unsqueeze(-2) * L + (1.0 - Mu.unsqueeze(-2)) * G)
    intensity = bg.unsqueeze(-1) + profile.sum(dim=-1)  # sum over peaks → [B, M]
    return intensity


def residuals(
    x: torch.Tensor,  # [B, 1 + 8P]
    z: torch.Tensor,  # [B, M] target
    Rs: torch.Tensor,  # [B, M]
    Etas: torch.Tensor,  # [B, M]
    pixel_mask: torch.Tensor,  # [B, M] {0, 1}
    n_peaks: int,
) -> torch.Tensor:
    """Per-pixel residual ``model - z``, zeroed where ``pixel_mask == 0``."""
    model = forward_pseudo_voigt(x, Rs, Etas, n_peaks)
    r = (model - z) * pixel_mask
    return r


def cost(
    x: torch.Tensor,
    z: torch.Tensor,
    Rs: torch.Tensor,
    Etas: torch.Tensor,
    pixel_mask: torch.Tensor,
    n_peaks: int,
) -> torch.Tensor:
    """Sum-of-squares per region. Returns shape ``[B]``."""
    r = residuals(x, z, Rs, Etas, pixel_mask, n_peaks)
    return (r * r).sum(dim=-1)


def integrated_intensity(
    x: torch.Tensor,
    Rs: torch.Tensor,
    Etas: torch.Tensor,
    pixel_mask: torch.Tensor,
    n_peaks: int,
):
    """Replicate ``calculateIntegratedIntensity`` in C (lines 781-840).

    For each pixel and each peak j, compute the pure peak intensity ``peak_j``
    (without bg). If ``peak_j > bg``, this pixel "belongs" to peak j: count
    it (``nrOfPixels[j] += 1``) and add ``bg + peak_j`` to its integrated
    intensity. Otherwise add only ``peak_j``.

    Returns: ``(integrated [B, P], n_pix [B, P])``.
    """
    bg, Imax, R, Eta, Mu, sgR, slR, sgE, slE = split_params(x, n_peaks)
    dR = Rs.unsqueeze(-1) - R.unsqueeze(-2)
    dE = Etas.unsqueeze(-1) - Eta.unsqueeze(-2)
    dR2 = dR * dR
    dE2 = dE * dE

    eps = torch.tensor(1e-12, dtype=x.dtype, device=x.device)
    invSGR2 = 1.0 / (sgR * sgR + eps).unsqueeze(-2)
    invSLR2 = 1.0 / (slR * slR + eps).unsqueeze(-2)
    invSGE2 = 1.0 / (sgE * sgE + eps).unsqueeze(-2)
    invSLE2 = 1.0 / (slE * slE + eps).unsqueeze(-2)

    L = 1.0 / ((dR2 * invSLR2 + 1.0) * (dE2 * invSLE2 + 1.0))
    G = torch.exp(-0.5 * (dR2 * invSGR2 + dE2 * invSGE2))
    peak_int = Imax.unsqueeze(-2) * (
        Mu.unsqueeze(-2) * L + (1.0 - Mu.unsqueeze(-2)) * G
    )  # [B, M, P]

    bg_exp = bg.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    over_bg = peak_int > bg_exp  # [B, M, P]
    pmask = pixel_mask.unsqueeze(-1)  # [B, M, 1]
    over_bg = over_bg & (pmask > 0)

    bg_to_add = torch.where(over_bg, bg_exp, torch.zeros_like(peak_int))
    integrated = (peak_int + bg_to_add) * pmask  # masked padding = 0
    integrated = integrated.sum(dim=-2)  # [B, P]
    n_pix = over_bg.sum(dim=-2)  # [B, P]
    return integrated, n_pix


__all__ = [
    "split_params",
    "forward_pseudo_voigt",
    "residuals",
    "cost",
    "integrated_intensity",
]

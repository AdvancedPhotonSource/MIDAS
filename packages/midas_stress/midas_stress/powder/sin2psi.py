"""sin²ψ method for residual stress from a powder ring.

Given a ring of fixed (h k l) on a 2D detector, the d-spacing extracted
at each azimuth ψ traces a straight line in ``sin²ψ``::

    d(ψ) - d_0
    ────────── = (s2/2) σ_φφ sin²ψ + s1 (σ_11 + σ_22 + σ_33)
       d_0

Standard textbook reference: Noyan & Cohen, *Residual Stress*. The
``X-ray elastic constants`` (XECs) ``s1, s2`` come from the single-
crystal stiffness via the Reuss / Voigt / Eshelby-Kröner average; this
stub uses the Reuss form (lower bound, tensile direction).

We expose:

- :func:`extract_d_vs_psi` — pull a per-ψ d-spacing array from a cake
  (2D η-vs-R) integration around one ring.
- :func:`fit_sin2psi` — linear regression d² vs sin²ψ → ε_φψ →
  σ_φφ via XEC inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Sin2PsiResult:
    sigma_phi_phi: float          # in-plane stress component (MPa or input units)
    epsilon_phi_phi_slope: float  # raw slope of ε(ψ²) before XEC inversion
    intercept: float
    rms_residual: float
    d0: float


def extract_d_vs_psi(
    int2d: np.ndarray,
    eta_axis_deg: np.ndarray,
    R_axis_to_d: np.ndarray,
    *,
    hkl_d0: float,
    capture_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-ψ d-spacing extraction from one ring of a cake.

    Parameters
    ----------
    int2d :
        ``(n_eta, n_R)`` integrated cake.
    eta_axis_deg :
        ``(n_eta,)`` η axis (degrees).
    R_axis_to_d :
        ``(n_R,)`` mapping from R-bin to d-spacing (Å). Caller is
        responsible for the conversion (use ``λ / (2 sin(θ))``).
    hkl_d0 :
        Nominal d-spacing of the ring of interest (Å).
    capture_radius :
        ±window in d-spacing around ``hkl_d0`` for centroid extraction.

    Returns
    -------
    psi_deg, d_spacing : both shape ``(n_eta,)``.
        ``psi_deg`` matches the η axis (we treat ψ ≡ |η - 90°| for an
        ω-axis horizontal-detector geometry; downstream code can remap
        if a different convention applies).
    """
    int2d = np.asarray(int2d, dtype=np.float64)
    eta = np.asarray(eta_axis_deg, dtype=np.float64)
    d_axis = np.asarray(R_axis_to_d, dtype=np.float64)
    if int2d.shape != (eta.shape[0], d_axis.shape[0]):
        raise ValueError(
            f"int2d shape {int2d.shape} != "
            f"(n_eta={eta.shape[0]}, n_R={d_axis.shape[0]})"
        )
    in_ring = np.abs(d_axis - hkl_d0) <= capture_radius
    if not in_ring.any():
        raise ValueError(
            f"no R bins in d ± {capture_radius} of d_0 = {hkl_d0}"
        )
    sub_d = d_axis[in_ring]                         # (m,)
    sub_I = int2d[:, in_ring]                       # (n_eta, m)
    # Intensity-weighted centroid d per η-bin
    weights = np.maximum(sub_I, 0.0)
    norm = weights.sum(axis=1)
    d_per_eta = np.where(
        norm > 0.0,
        (weights * sub_d[None, :]).sum(axis=1) / np.where(norm > 0, norm, 1),
        np.nan,
    )
    psi_deg = np.abs(eta - 90.0)
    return psi_deg, d_per_eta


def fit_sin2psi(
    psi_deg: np.ndarray,
    d_spacing: np.ndarray,
    d0: float,
    *,
    s1: Optional[float] = None,
    s2: Optional[float] = None,
    triaxial_term: float = 0.0,
) -> Sin2PsiResult:
    """Linear regression of d(ψ²) and conversion to σ via XECs.

    The XEC formulation::

        ε_φψ = (s2/2) σ_φφ sin²ψ + s1 (σ_11 + σ_22 + σ_33)

    Caller supplies XECs ``s1, s2`` derived from the stiffness via
    Reuss / Voigt averaging. If both are None, only the slope of ε(ψ²)
    is reported and ``sigma_phi_phi`` is returned as 0.0 (pure-strain
    output).

    The ``triaxial_term`` is the assumed value of ``σ_11 + σ_22 +
    σ_33`` — set to 0 for plane-stress (typical surface analysis).
    """
    psi = np.asarray(psi_deg, dtype=np.float64)
    d = np.asarray(d_spacing, dtype=np.float64)
    valid = np.isfinite(d)
    psi = psi[valid]
    d = d[valid]
    if d.size < 4:
        raise ValueError("need >= 4 valid d-spacing points for sin²ψ fit")
    sin2psi = np.sin(np.deg2rad(psi)) ** 2
    epsilon = (d - d0) / d0
    # Linear regression ε = m sin²ψ + b
    m, b = np.polyfit(sin2psi, epsilon, 1)
    fit = m * sin2psi + b
    rms = float(np.sqrt(np.mean((epsilon - fit) ** 2)))
    if s1 is None or s2 is None:
        return Sin2PsiResult(0.0, float(m), float(b), rms, float(d0))
    # σ_φφ from slope: m = (s2/2) σ_φφ → σ_φφ = 2m/s2
    sigma_phi = 2.0 * m / s2 - 2.0 * (s1 / s2) * triaxial_term
    return Sin2PsiResult(float(sigma_phi), float(m), float(b), rms, float(d0))


__all__ = ["extract_d_vs_psi", "fit_sin2psi", "Sin2PsiResult"]

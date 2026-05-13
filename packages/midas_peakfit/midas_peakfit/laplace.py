"""Laplace approximation at MAP — cheap Bayesian uncertainty.

Compute the Hessian H of the negative log-posterior at the MAP estimate;
the posterior is approximated as N(MAP, H⁻¹).  Marginal 1σ per parameter is
``sqrt(diag(H⁻¹))``.

Cheap because we only do one Hessian build at convergence.  Adequate when
the posterior is Gaussian-like (typical for calibration parameters with
informative data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from midas_peakfit.pack import (
    refined_indices, refined_bounds,
    write_refined_back, unpack_spec, pack_spec,
)
from midas_peakfit.spec import ParameterSpec


@dataclass
class LaplaceResult:
    map_unpacked: Dict[str, torch.Tensor]
    refined_names: List[str]
    refined_offsets: List[int]
    refined_sizes: List[int]
    map_refined: torch.Tensor
    cov: torch.Tensor                 # [N_ref, N_ref]
    sigma_per_dim: torch.Tensor       # [N_ref]; √diag(cov)


def laplace_at_map(
    spec: ParameterSpec,
    nll_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    map_unpacked: Dict[str, torch.Tensor],
    *,
    fallback_span: float = 1.0,
    ridge: float = 1e-9,
    dtype=torch.float64, device="cpu",
) -> LaplaceResult:
    """Compute Laplace covariance at the supplied MAP unpacked dict.

    The Hessian is computed in *unbounded* u-space so the Gaussian
    approximation respects the parameter bounds; we then transform back to
    bounded x-space via the local Jacobian for reporting.

    Parameters
    ----------
    nll_fn : callable
        Negative log-posterior (data NLL + prior NLL).
    map_unpacked : dict
        Unpacked dict from a converged optimisation.
    """
    from midas_peakfit.reparam import x_to_u, u_to_x

    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    # Build x_ref from MAP dict (overrides spec init).
    x_full_map = x_full.clone()
    for name, val in map_unpacked.items():
        sl = info.slice(name)
        x_full_map[sl] = val.detach().to(dtype=dtype, device=device).reshape(-1)
    x_ref_map = x_full_map.index_select(0, refined_idx)
    u_map = x_to_u(x_ref_map, lo, hi)

    def nll_of_u(u: torch.Tensor) -> torch.Tensor:
        x_ref = u_to_x(u, lo, hi)
        x_full_now = write_refined_back(x_full_map, x_ref, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        return nll_fn(unpacked)

    H = torch.autograd.functional.hessian(nll_of_u, u_map.detach().clone())
    H = 0.5 * (H + H.transpose(-1, -2))   # symmetrize numerical noise

    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    H_reg = H + ridge * eye
    cov_u = torch.linalg.inv(H_reg)

    # Transform covariance from u-space to x-space via local Jacobian
    # dx/du = (hi - lo) σ(u) (1 - σ(u)).
    s = torch.sigmoid(u_map)
    span = (hi - lo)
    dxdu = span * s * (1.0 - s)
    cov_x = cov_u * dxdu.unsqueeze(0) * dxdu.unsqueeze(1)

    sigma = torch.sqrt(torch.diag(cov_x).clamp(min=0.0))

    refined_names: List[str] = []
    refined_offsets: List[int] = []
    refined_sizes: List[int] = []
    cur = 0
    for n, o, s_, r_ in zip(info.names, info.offsets, info.sizes, info.refined):
        if r_:
            refined_names.append(n)
            refined_offsets.append(cur)
            refined_sizes.append(s_)
            cur += s_

    return LaplaceResult(
        map_unpacked=map_unpacked,
        refined_names=refined_names,
        refined_offsets=refined_offsets,
        refined_sizes=refined_sizes,
        map_refined=x_ref_map.detach(),
        cov=cov_x,
        sigma_per_dim=sigma,
    )


def fisher_at_map(
    spec: ParameterSpec,
    residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    map_unpacked: Dict[str, torch.Tensor],
    *,
    sigma_r,
    fallback_span: float = 1.0,
    ridge: float = 1e-9,
    dtype=torch.float64, device="cpu",
) -> LaplaceResult:
    """Fisher-information / J'J approximation to the Laplace covariance.

    Builds a single Jacobian via :func:`torch.func.jacrev` instead of the
    full Hessian.  For the Gauss-Newton model

        NLL(θ) = 0.5 / σ_r² · Σ r(θ)²

    the Fisher info is ``F = J'J / σ_r²``, and ``Cov ≈ F⁻¹``.  Same
    leading behaviour as the full Hessian when the model is well-fit
    (residuals close to Gaussian noise of stddev σ_r), but **O(N) instead
    of O(N²)** in parameter count.

    Critical for high-DOF problems: Pilatus has 240 panel DOFs + 20
    geometry/distortion → full ``hessian()`` is ~67k function evals
    (>30 min).  This routine replaces that with one ``jacrev`` call
    (typically <30 s for the same problem).

    Pass ``residual_fn`` (a closure mapping ``unpacked → r [M]``) and
    ``sigma_r`` (the residual stddev at MAP).

    ``sigma_r`` may be a Python float (homoscedastic noise on every
    residual row, the historical default) OR a 1-D ``torch.Tensor`` of
    length ``M`` carrying the per-row noise.  The tensor form is required
    when the residual closure concatenates contributions with different
    noise scales — e.g. data rows scaled by the empirical strain noise
    plus Gaussian-prior rows that are already unit-normalised by their
    prior stddev.  If the lengths disagree the function raises.
    """
    from midas_peakfit.reparam import x_to_u, u_to_x
    from torch.func import jacfwd, jacrev

    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    x_full_map = x_full.clone()
    for name, val in map_unpacked.items():
        sl = info.slice(name)
        x_full_map[sl] = val.detach().to(dtype=dtype, device=device).reshape(-1)
    x_ref_map = x_full_map.index_select(0, refined_idx)
    u_map = x_to_u(x_ref_map, lo, hi)

    def r_of_u(u: torch.Tensor) -> torch.Tensor:
        x_ref = u_to_x(u, lo, hi)
        x_full_now = write_refined_back(x_full_map, x_ref, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        return residual_fn(unpacked)

    # Single Jacobian: J[m, n] = ∂r_m / ∂u_n.  Use forward-mode AD when
    # N_params << N_residuals (typical for calibration: M ~ 1k–4k fits,
    # N ~ 5–260 params).  jacfwd does N forward passes; jacrev does M
    # backward passes.  For headline geometry (N=20, M=4500), jacfwd is
    # ~50× faster.  For full Pilatus (N=260, M=2935), jacfwd is still
    # ~10× faster.
    n_params = int(u_map.numel())
    m_residuals_estimate = int(r_of_u(u_map.detach().clone()).numel())
    if n_params < m_residuals_estimate:
        J = jacfwd(r_of_u)(u_map.detach().clone())
    else:
        J = jacrev(r_of_u)(u_map.detach().clone())
    # Per-row noise: build inv-variance vector, weight Jacobian rows so
    # that the resulting J^T·diag(1/σ²)·J has each row contributing
    # 1/σ_r[m]² instead of a uniform 1/σ_r².  Backwards-compatible
    # scalar path (the 99 % case) keeps the original code path.
    if isinstance(sigma_r, torch.Tensor) and sigma_r.numel() > 1:
        if sigma_r.numel() != J.shape[0]:
            raise ValueError(
                f"per-row sigma_r has length {sigma_r.numel()} but the "
                f"residual closure returned {J.shape[0]} rows; the lengths "
                f"must match"
            )
        inv_sigma = (1.0 / sigma_r.to(dtype=J.dtype, device=J.device)).unsqueeze(1)
        Jw = J * inv_sigma                       # row-wise weighting
        F_u = Jw.transpose(0, 1) @ Jw
    else:
        sigma_r_scalar = float(sigma_r if not isinstance(sigma_r, torch.Tensor)
                                else sigma_r.item())
        JtJ = J.transpose(0, 1) @ J
        F_u = JtJ / (sigma_r_scalar ** 2)
    eye = torch.eye(F_u.shape[0], dtype=F_u.dtype, device=F_u.device)
    F_reg = F_u + ridge * eye
    # Robust pseudo-inverse for rank-deficient Fishers: many calibration
    # problems have parameters that the data cannot constrain (e.g.
    # individual panel deltas under single-image rotation-around-beam
    # gauge degeneracy).  ``torch.linalg.inv`` throws on a zero pivot;
    # ``torch.linalg.pinv`` returns the Moore-Penrose pseudoinverse, which
    # gives finite (large) σ for rank-deficient directions instead.
    try:
        cov_u = torch.linalg.inv(F_reg)
    except torch._C._LinAlgError:
        cov_u = torch.linalg.pinv(F_reg)

    # Transform u-space cov to x-space via dx/du = (hi-lo) σ(u)(1-σ(u)).
    s = torch.sigmoid(u_map)
    dxdu = (hi - lo) * s * (1.0 - s)
    cov_x = cov_u * dxdu.unsqueeze(0) * dxdu.unsqueeze(1)
    sigma = torch.sqrt(torch.diag(cov_x).clamp(min=0.0))

    refined_names: List[str] = []
    refined_offsets: List[int] = []
    refined_sizes: List[int] = []
    cur = 0
    for n, o, s_, r_ in zip(info.names, info.offsets, info.sizes, info.refined):
        if r_:
            refined_names.append(n)
            refined_offsets.append(cur)
            refined_sizes.append(s_)
            cur += s_

    return LaplaceResult(
        map_unpacked=map_unpacked,
        refined_names=refined_names,
        refined_offsets=refined_offsets,
        refined_sizes=refined_sizes,
        map_refined=x_ref_map.detach(),
        cov=cov_x,
        sigma_per_dim=sigma,
    )


def report_laplace(res: LaplaceResult) -> str:
    """Pretty-print marginal 1σ per parameter."""
    lines = ["Laplace marginals (MAP ± 1σ):"]
    for n, o, s in zip(res.refined_names, res.refined_offsets, res.refined_sizes):
        for k in range(s):
            i = o + k
            mean = float(res.map_refined[i])
            std = float(res.sigma_per_dim[i])
            tag = f"{n}[{k}]" if s > 1 else n
            lines.append(f"  {tag:<24s} {mean:+.6e}  ±  {std:.3e}")
    return "\n".join(lines)


__all__ = ["LaplaceResult", "laplace_at_map", "fisher_at_map", "report_laplace"]

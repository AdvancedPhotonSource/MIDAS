"""Soft constraints for ill-conditioned calibration problems.

The canonical case is the gauge fix on per-panel shifts: a uniform
shift of all panels is mathematically degenerate with the global beam
center, so without a constraint the LM has a zero-eigenvalue mode in
that direction.  v1 fixes one panel's deltas to zero (the
``fix_panel_id`` mask in :mod:`midas_calibrate_v2.forward.panels`).
v2 also supports a softer **Σ panel = 0** penalty, which is the gauge
choice used in Wright, Giacobbe & Lawrence Bright (2022, *Crystals*
12(2), 255 §3.2).  The Σ = 0 gauge is symmetric across panels — no
"special" reference panel — and is more numerically robust because
the constraint contribution to the Fisher matrix is exactly along the
nullspace, leaving the data-determined directions untouched.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def zero_sum_residual(
    unpacked: dict,
    *,
    block_names: Sequence[str] = (
        "panel_delta_yz", "panel_delta_theta",
        "panel_delta_lsd", "panel_delta_p2",
        "delta_r_k",
    ),
    lambda_zs: float = 1e6,
) -> torch.Tensor:
    """Quadratic residual penalty: ``sqrt(λ) × Σ_panel block``.

    Concatenate the returned tensor onto the data residual so the LM
    sees these as additional rows of the residual vector.  The squared
    norm contribution to the loss is ``λ × Σ block²`` per DOF;
    ``λ → ∞`` enforces ``Σ block = 0`` hard, while finite ``λ`` makes it
    a Gaussian prior of stddev ``1/sqrt(λ)``.

    Default ``λ = 1e6``: with a panel-shift bound of 4 px, a fully
    saturated gauge mode (all panels shifted by 4 px) would contribute
    ``sqrt(λ) × 4 = 4000`` to the residual, which dwarfs the
    data-residual scale (per-fit strain ~1e-5).  The LM will therefore
    drive ``Σ → 0`` to numerical precision, and the gauge mode's σ
    becomes ``1/sqrt(λ)`` rather than ridge-floor.

    Parameters
    ----------
    unpacked
        Output of :func:`unpack_spec`; keys = parameter names.
    block_names
        Per-panel parameter blocks to constrain.  Missing keys are
        silently ignored.
    lambda_zs
        Penalty weight.  Make this large; default ``1e6`` is fine for
        most cases.

    Returns
    -------
    A 1-D residual tensor whose length is the total flattened size of
    the per-block sums (e.g. 5 for the standard Pilatus block:
    yz=2 + θ=1 + Lsd=1 + p2=1).
    """
    pieces = []
    sqrt_lam = math.sqrt(lambda_zs)
    ref = None
    for nm in block_names:
        if nm not in unpacked:
            continue
        block = unpacked[nm]
        if block.numel() == 0:
            continue
        if ref is None:
            ref = block
        # Sum across the panel axis (axis 0).  The remaining axes (e.g.
        # the (δy, δz) DOF axis for panel_delta_yz) are kept and
        # concatenated, so the LM sees one residual per DOF per block.
        s = block.sum(dim=0).flatten()
        pieces.append(sqrt_lam * s)
    if not pieces:
        # Pull dtype/device from any value in unpacked.
        if ref is None:
            for v in unpacked.values():
                if isinstance(v, torch.Tensor):
                    ref = v
                    break
        if ref is None:
            return torch.zeros(0, dtype=torch.float64)
        return torch.zeros(0, dtype=ref.dtype, device=ref.device)
    return torch.cat(pieces)


def gaussian_prior_residual(
    unpacked: dict,
    spec,
) -> torch.Tensor:
    """Concatenate per-parameter Gaussian-prior residuals.

    For each parameter ``θ`` whose ``Parameter.prior`` is a
    :class:`~midas_calibrate_v2.parameters.parameter.GaussianPrior` with
    mean ``μ`` and stddev ``σ``, append the row ``(θ - μ) / σ``.  Vector
    parameters contribute one row per element (the prior is broadcast as
    independent Gaussians per element).

    LM minimises the sum-of-squares of the residual vector; appending
    these rows therefore adds ``Σ (θ - μ)² / σ²`` to the cost, which is
    exactly ``-2 log p(θ)`` for a Gaussian prior up to a parameter-
    independent constant.  This is what makes refining ``pxY``/``pxZ``
    on a single image well-conditioned (otherwise the
    ``(L_sd, p_x)`` multiplicative gauge is rank-deficient — see the
    pixel-size identifiability analysis in the paper).

    Parameters with non-Gaussian priors (e.g. :class:`HalfCauchyPrior`)
    are silently skipped: those need MCMC / VI rather than a least-
    squares residual.  Parameters with no prior are also skipped.

    Returns
    -------
    A 1-D residual tensor (possibly empty) with one entry per scalar
    DOF that carries a Gaussian prior.
    """
    from midas_peakfit.parameter import GaussianPrior   # local to avoid cycle

    pieces = []
    ref = None
    for name, param in spec.parameters.items():
        if param.prior is None or not isinstance(param.prior, GaussianPrior):
            continue
        if name not in unpacked:
            continue
        x = unpacked[name]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float64)
        if ref is None:
            ref = x
        mu = float(param.prior.mean)
        sd = float(param.prior.std)
        if sd <= 0.0:
            continue
        # Per-element residual (vector params broadcast naturally).
        pieces.append(((x - mu) / sd).flatten())

    if not pieces:
        if ref is None:
            for v in unpacked.values():
                if isinstance(v, torch.Tensor):
                    ref = v
                    break
        if ref is None:
            return torch.zeros(0, dtype=torch.float64)
        return torch.zeros(0, dtype=ref.dtype, device=ref.device)
    return torch.cat(pieces)


__all__ = ["zero_sum_residual", "gaussian_prior_residual"]

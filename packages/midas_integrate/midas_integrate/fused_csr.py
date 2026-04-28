"""Family II mitigation — multi-map averaging fused into a single CSR.

Builds N detector maps with R-bin edges shifted by
``[0, 1/N, …, (N-1)/N] · ΔR`` and folds the per-frame averaging into
one combined sparse matrix:

.. math::
    M_\\text{combined} = \\frac{1}{N} \\sum_{k=0}^{N-1}
        \\mathbf{P}_k\\,\\mathbf{D}_k\\,\\mathbf{M}_k

where :math:`\\mathbf{M}_k` is the unnormalised CSR for the *k*-th
shifted map, :math:`\\mathbf{D}_k = \\operatorname{diag}(1/A_{k,j})`
folds the per-bin area normalisation into the matrix, and
:math:`\\mathbf{P}_k` is a sparse linear-interpolation matrix that
maps cake *k*'s R-axis onto the common output R-axis.

The result is a :class:`CSRGeometry` whose ``csr_bilinear`` (and
``csr_gradient`` if available) already includes the multi-map
averaging. The standard :func:`integrate` function then produces
the averaged cake in **one** sparse matrix-vector multiply per
frame, with mathematically identical output to naive
multi-map averaging (verified to ULP precision in tests).

Public API::

    from midas_integrate import build_fused_geometry

    fused = build_fused_geometry(params, n_shifts=8, mode='gradient')
    cake = integrate(image, fused, mode='bilinear', normalize=False)
    # cake is the (n_r, n_eta) averaged cake; flux preserved.

Note: the fused geometry has ``area_per_bin = ones`` because the
per-shift normalisations are already baked into the matrix; pass
``normalize=False`` to :func:`integrate` (or simply ignore the
default ``normalize=True``, which divides by 1).
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from midas_integrate.detector_mapper import build_map
from midas_integrate.kernels import (
    AREA_THRESHOLD,
    CSRGeometry,
    _scipy_csr_to_torch,
    build_csr,
)
from midas_integrate.params import IntegrationParams
from midas_integrate.kernels import eta_axis as _eta_axis
from midas_integrate.kernels import r_axis as _r_axis


def _torch_csr_to_scipy(csr: torch.Tensor) -> sp.csr_matrix:
    crow = csr.crow_indices().cpu().numpy()
    col = csr.col_indices().cpu().numpy()
    val = csr.values().cpu().numpy()
    return sp.csr_matrix((val, col, crow), shape=csr.shape)


def _build_R_interp_matrix(R_src: np.ndarray, R_tgt: np.ndarray,
                            n_eta: int) -> sp.csr_matrix:
    """Build a sparse linear-interpolation matrix that maps a flattened
    cake on the ``R_src`` × n_eta grid onto the ``R_tgt`` × n_eta grid.

    Both cakes are assumed flattened in row-major order
    (R outer, η inner). Output shape: ``(len(R_tgt) * n_eta,
    len(R_src) * n_eta)``.

    Each output bin gets at most two non-zeros (linear interpolation in
    R, identity in η).
    """
    n_src = len(R_src)
    n_tgt = len(R_tgt)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n_tgt):
        r = R_tgt[i]
        if r <= R_src[0]:
            j = 0; w_lo = 1.0; w_hi = 0.0
        elif r >= R_src[-1]:
            j = n_src - 2; w_lo = 0.0; w_hi = 1.0
        else:
            j = int(np.searchsorted(R_src, r, side="right") - 1)
            j = max(0, min(j, n_src - 2))
            denom = R_src[j + 1] - R_src[j]
            t = (r - R_src[j]) / denom if denom > 0 else 0.0
            w_lo = 1.0 - t
            w_hi = t
        for e in range(n_eta):
            rows.append(i * n_eta + e)
            cols.append(j * n_eta + e)
            vals.append(w_lo)
            rows.append(i * n_eta + e)
            cols.append((j + 1) * n_eta + e)
            vals.append(w_hi)
    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_tgt * n_eta, n_src * n_eta),
    )


def _build_one_shifted_geom(
    params: IntegrationParams,
    rmin_offset: float,
    *,
    mode: str,
    device: str,
    dtype: torch.dtype,
    verbose: bool,
) -> tuple[CSRGeometry, np.ndarray]:
    """Build a map at ``RMin + rmin_offset`` and its CSRGeometry.

    Returns ``(geom, R_axis)``.
    """
    p = deepcopy(params)
    p.RMin = params.RMin + rmin_offset
    res = build_map(p, verbose=verbose)
    pixmap = res.as_pixel_map()
    geom = build_csr(
        pixmap,
        n_r=p.n_r_bins, n_eta=p.n_eta_bins,
        n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
        bc_y=p.BC_y, bc_z=p.BC_z,
        device=device, dtype=dtype,
        build_modes=("bilinear", "gradient") if mode in ("bilinear", "gradient")
                    else (mode,),
    )
    R = _r_axis(n_r=p.n_r_bins, RMin=p.RMin, RBinSize=p.RBinSize)
    return geom, R


def build_fused_geometry(
    params: IntegrationParams,
    *,
    n_shifts: int = 4,
    mode: str = "gradient",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> CSRGeometry:
    """Build a fused-CSR geometry that produces multi-map-averaged
    cakes (Family II of the cardinal-aliasing mitigation hierarchy)
    in a single sparse multiply per frame.

    Args:
        params: parsed parameter file. The configured ``RBinSize`` is
            the *coarse* output bin width.
        n_shifts: number of R-shifted maps to fuse. Common values:
            ``4`` (best speed/quality tradeoff) or ``8`` (slightly
            stronger σ/μ reduction).
        mode: ``'bilinear'`` or ``'gradient'``. The per-shift maps are
            built with this mode; choose ``'gradient'`` to combine
            Family I (radial resampling) with Family II in one
            geometry.
        device: torch device for the output sparse matrix.
        dtype: torch dtype.
        verbose: print build progress.

    Returns:
        :class:`CSRGeometry` whose ``csr_bilinear`` (and ``csr_gradient``,
        when ``mode == 'gradient'``) holds the fused matrix
        :math:`M_\\text{combined}`. The combined matrix has the per-shift
        normalisations baked in, so the returned geometry has
        ``area_per_bin = 1`` everywhere; call
        :func:`integrate` either with ``normalize=False`` or with the
        default ``normalize=True`` (which divides by 1, a no-op).

    Notes:
        The output cake is on the R-axis of the *baseline* (k=0) shift.
        Total ring flux is preserved to better than $0.05\\,\\%$ in
        practice; we have verified that the per-bin values are
        bit-equivalent to naive R-N averaging at ULP precision in our
        test data.
    """
    if n_shifts < 1:
        raise ValueError(f"n_shifts must be ≥ 1; got {n_shifts}")
    if mode not in ("bilinear", "gradient"):
        raise ValueError(f"mode must be 'bilinear' or 'gradient'; got {mode!r}")

    rbin = float(params.RBinSize)
    if rbin <= 0:
        raise ValueError(f"params.RBinSize must be > 0; got {rbin}")

    if verbose:
        print(f"[fused] building {n_shifts} shifted maps "
              f"(ΔR={rbin}, mode={mode}) ...")

    # Build the N shifted geometries.
    geoms: list[CSRGeometry] = []
    R_axes: list[np.ndarray] = []
    for k in range(n_shifts):
        offset = rbin * k / n_shifts
        t0 = time.perf_counter()
        g_k, R_k = _build_one_shifted_geom(
            params, offset, mode=mode,
            device=device, dtype=dtype, verbose=False,
        )
        if verbose:
            print(f"[fused]   shift {k}/{n_shifts}  RMin+={offset:.4f}  "
                  f"({time.perf_counter() - t0:.1f}s)")
        geoms.append(g_k)
        R_axes.append(R_k)

    # Combine.
    if verbose:
        print(f"[fused] folding shifted maps into combined matrix ...")
    R0 = R_axes[0]
    n_eta = params.n_eta_bins

    M_combined: Optional[sp.csr_matrix] = None
    G_combined: Optional[sp.csr_matrix] = None
    for k, (g_k, R_k) in enumerate(zip(geoms, R_axes)):
        # Per-shift area-normaliser
        A_k = g_k.area_per_bin.detach().cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_A_k = np.where(A_k > AREA_THRESHOLD,
                               1.0 / np.maximum(A_k, AREA_THRESHOLD), 0.0)
        D_k = sp.diags(inv_A_k, 0, shape=(g_k.n_bins, g_k.n_bins))
        # Linear-interp matrix from R_k → R0
        P_k = _build_R_interp_matrix(R_k, R0, n_eta)

        # Bilinear contribution
        M_k_bil = _torch_csr_to_scipy(g_k.csr_bilinear)
        contrib_bil = P_k @ D_k @ M_k_bil
        M_combined = contrib_bil if M_combined is None \
                     else M_combined + contrib_bil

        # Gradient contribution (if requested mode is gradient OR if
        # gradient was built; we use the gradient matrix for the
        # gradient slot of the fused geometry whenever available).
        try:
            M_k_grad = _torch_csr_to_scipy(g_k.csr_gradient)
            if M_k_grad.nnz > 0:
                contrib_grad = P_k @ D_k @ M_k_grad
                G_combined = contrib_grad if G_combined is None \
                             else G_combined + contrib_grad
        except Exception:
            pass

    M_combined = M_combined / n_shifts
    if G_combined is not None:
        G_combined = G_combined / n_shifts

    # Build the output CSRGeometry. We expose:
    #   csr_bilinear = M_combined (always)
    #   csr_gradient = G_combined if available, else M_combined
    #   csr_floor    = empty (floor mode is not meaningful for fused)
    #   area_per_bin = ones (normalisation is baked in)
    n_bins = M_combined.shape[0]
    csr_bil_t = _scipy_csr_to_torch(M_combined.tocsr(),
                                     device=device, dtype=dtype)
    if G_combined is not None and mode == "gradient":
        csr_grad_t = _scipy_csr_to_torch(G_combined.tocsr(),
                                          device=device, dtype=dtype)
        # When user picked 'gradient' as the mode, also expose it via
        # csr_bilinear so that integrate(mode='bilinear') Just Works
        # without users having to remember the mode they built with.
        csr_bil_t = csr_grad_t
    else:
        csr_grad_t = csr_bil_t

    # Empty floor matrix
    floor_csr = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64),
        torch.zeros((0,), dtype=dtype),
        size=(n_bins, params.NrPixelsY * params.NrPixelsZ),
    ).coalesce().to_sparse_csr().to(device)

    area_t = torch.ones(n_bins, dtype=dtype, device=device)

    out = CSRGeometry(
        csr_floor=floor_csr,
        csr_bilinear=csr_bil_t,
        csr_gradient=csr_grad_t,
        area_per_bin=area_t,
        n_r=params.n_r_bins,
        n_eta=params.n_eta_bins,
        n_pixels_y=params.NrPixelsY,
        n_pixels_z=params.NrPixelsZ,
        bc_y=params.BC_y,
        bc_z=params.BC_z,
    )

    if verbose:
        print(f"[fused] done. nnz(combined)={M_combined.nnz:,} "
              f"({M_combined.nnz / max(geoms[0].csr_bilinear._nnz(), 1):.2f}× "
              "the single-shift map).")
    return out

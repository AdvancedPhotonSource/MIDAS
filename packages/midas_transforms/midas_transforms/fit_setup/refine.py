"""5-parameter geometry refine — delegates to ``midas_calibrate.refine_geometry``.

This replaces the C ``FitTiltBCLsd`` NLopt Nelder-Mead fit with the LM solver
(plus optional ADAM fallback) shipped by ``midas-calibrate``. Per
``dev/implementation_plan.md`` §3.2 / §4.3 / §8 risk #1, no NLopt and no
Nelder-Mead are used.

The fit minimises mean per-eta-bin strain (|1 - Rcorr/RIdeal|) — the same
objective as the C reference (``FitSetupParamsAllZarr.c:200-266``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class RefineParams:
    Lsd: float
    BC_y: float
    BC_z: float
    ty: float
    tz: float
    mean_strain_uE: float
    rc: int


def refine_5param(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    ring_idx: torch.Tensor,
    ring_d_spacing: torch.Tensor,
    ring_two_theta_deg: torch.Tensor,
    *,
    Lsd: float,
    BC_y: float,
    BC_z: float,
    tx: float,
    ty: float,
    tz: float,
    p_coeffs: Tuple[float, ...],   # length 15
    px: float,
    rho_d: float,
    tol_lsd: float = 5000.0,
    tol_bc: float = 1.0,
    tol_tilts: float = 1.0,
    max_iter: int = 200,
    device: str = "cpu",
    dtype=torch.float64,
) -> RefineParams:
    """Refine ``(Lsd, BC_y, BC_z, ty, tz)`` against the supplied per-spot
    pixel coordinates / ring assignments via ``midas_calibrate.refine_geometry``.

    The wrapping is direct: the inputs map 1:1 to ``CalibrationParams``
    fields, and the per-spot points become a list of ``FittedPoint``.
    """
    from midas_calibrate import (
        CalibrationParams, FittedPoint, RefineResult, build_ring_table,
        refine_geometry,
    )
    from midas_calibrate.rings import RingTable

    # Build CalibrationParams from current geometry. We hold ``tx`` fixed
    # at the input value, since the C ``FitTiltBCLsd`` does not refine ``tx``.
    cp = CalibrationParams()
    cp.Lsd = float(Lsd)
    cp.BC_y = float(BC_y)
    cp.BC_z = float(BC_z)
    cp.tx = float(tx)
    cp.ty = float(ty)
    cp.tz = float(tz)
    for i in range(15):
        setattr(cp, f"p{i}", float(p_coeffs[i]))
    cp.pxY = float(px)
    cp.pxZ = float(px)
    cp.RhoD = float(rho_d)
    # Refinement bounds (C-equivalent: Lsd ± tol_lsd, BC ± tol_bc, ty/tz ± tol_tilts).
    cp.Refine = {
        "Lsd": True, "BC_y": True, "BC_z": True, "ty": True, "tz": True,
    }
    cp.Bounds = {
        "Lsd": (Lsd - tol_lsd, Lsd + tol_lsd),
        "BC_y": (BC_y - tol_bc, BC_y + tol_bc),
        "BC_z": (BC_z - tol_bc, BC_z + tol_bc),
        "ty": (ty - tol_tilts, ty + tol_tilts),
        "tz": (tz - tol_tilts, tz + tol_tilts),
    }

    # Build a minimal RingTable (only d/two_theta needed).
    rt = RingTable(
        d_spacing=ring_d_spacing.detach().cpu().numpy(),
        two_theta_deg=ring_two_theta_deg.detach().cpu().numpy(),
        ring_numbers=list(range(1, 1 + len(ring_d_spacing))),
    )

    # Per-spot fitted points.
    Yn = Y_pix.detach().cpu().numpy()
    Zn = Z_pix.detach().cpu().numpy()
    Rn = ring_idx.detach().cpu().numpy()
    fits: List[FittedPoint] = [
        FittedPoint(Y_pix=float(y), Z_pix=float(z), ring_idx=int(r), snr=1.0)
        for y, z, r in zip(Yn, Zn, Rn)
    ]

    result: RefineResult = refine_geometry(
        params=cp, rt=rt, fits=fits,
        device=device, dtype=dtype, max_iter=max_iter,
    )

    return RefineParams(
        Lsd=result.params.Lsd,
        BC_y=result.params.BC_y,
        BC_z=result.params.BC_z,
        ty=result.params.ty,
        tz=result.params.tz,
        mean_strain_uE=result.mean_strain_uE,
        rc=result.rc,
    )

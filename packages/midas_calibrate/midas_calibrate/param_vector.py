"""Pack/unpack between CalibrationParams and a flat torch parameter vector.

Layout (length 23):
    [0]    Lsd
    [1, 2] BC_y, BC_z
    [3, 4] ty, tz
    [5..19] p0..p14
    [20]   Parallax
    [21]   Wavelength
    [22]   tx (fixed)
"""
from __future__ import annotations

from typing import Tuple

import torch

from .params import CalibrationParams

N_GEOM = 23


def pack(params: CalibrationParams, dtype=torch.float64, device="cpu") -> torch.Tensor:
    x = torch.zeros(N_GEOM, dtype=dtype, device=device)
    x[0] = params.Lsd
    x[1] = params.BC_y
    x[2] = params.BC_z
    x[3] = params.ty
    x[4] = params.tz
    for i in range(15):
        x[5 + i] = getattr(params, f"p{i}")
    x[20] = params.Parallax
    x[21] = params.Wavelength
    x[22] = params.tx
    return x


def unpack(x: torch.Tensor, params: CalibrationParams) -> CalibrationParams:
    """Mutate `params` in place from a flat tensor and return it."""
    x = x.detach().cpu()
    params.Lsd = float(x[0])
    params.BC_y = float(x[1])
    params.BC_z = float(x[2])
    params.ty = float(x[3])
    params.tz = float(x[4])
    for i in range(15):
        setattr(params, f"p{i}", float(x[5 + i]))
    params.Parallax = float(x[20])
    params.Wavelength = float(x[21])
    params.tx = float(x[22])
    return params


def bounds(params: CalibrationParams, dtype=torch.float64, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (lo, hi) tensors per the tolerance fields in `params`."""
    lo = torch.empty(N_GEOM, dtype=dtype, device=device)
    hi = torch.empty(N_GEOM, dtype=dtype, device=device)

    lo[0] = params.Lsd - params.tolLsd
    hi[0] = params.Lsd + params.tolLsd
    lo[1] = params.BC_y - params.tolBC; hi[1] = params.BC_y + params.tolBC
    lo[2] = params.BC_z - params.tolBC; hi[2] = params.BC_z + params.tolBC
    lo[3] = params.ty - params.tolTilts; hi[3] = params.ty + params.tolTilts
    lo[4] = params.tz - params.tolTilts; hi[4] = params.tz + params.tolTilts
    for i in range(15):
        cur = getattr(params, f"p{i}")
        lo[5 + i] = cur - params.tolDistortion
        hi[5 + i] = cur + params.tolDistortion
    lo[20] = params.Parallax - params.tolParallax; hi[20] = params.Parallax + params.tolParallax
    lo[21] = params.Wavelength - params.tolWavelength; hi[21] = params.Wavelength + params.tolWavelength
    # tx is held fixed: very tight bound around current value.
    lo[22] = params.tx - 1e-9; hi[22] = params.tx + 1e-9
    return lo, hi


def refine_mask(params: CalibrationParams) -> torch.Tensor:
    """Boolean [N_GEOM] mask of which parameters are refined.

    Parameters that are *not* refined get bounds collapsed to a tight ε around
    their current value, effectively freezing them.  This is the simplest way
    to use the generic LM with selective refinement.
    """
    m = torch.zeros(N_GEOM, dtype=torch.bool)
    m[0] = params.Refine.get("Lsd", True)
    m[1] = params.Refine.get("BC", True)
    m[2] = params.Refine.get("BC", True)
    m[3] = params.Refine.get("ty", True)
    m[4] = params.Refine.get("tz", True)
    for i in range(15):
        m[5 + i] = params.Refine.get(f"p{i}", True)
    m[20] = params.Refine.get("Parallax", False)
    m[21] = params.Refine.get("Wavelength", False)
    m[22] = False  # tx never refined
    return m


def collapse_unrefined_bounds(lo: torch.Tensor, hi: torch.Tensor, x0: torch.Tensor,
                               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """For non-refined params, set lo=hi=x0 (within ε to keep the sigmoid finite).

    The reparameterization in midas_peakfit.reparam clamps to (lo, hi); collapsing
    the range freezes the parameter at x0.
    """
    eps = 1e-9
    lo_out = torch.where(mask, lo, x0 - eps)
    hi_out = torch.where(mask, hi, x0 + eps)
    return lo_out, hi_out

"""Torch counterparts of the metric-tensor / d-spacing helpers in lattice.py.

Imports torch lazily; torch is an optional dependency.  Public functions
accept either a 6-tuple/array (a,b,c,α,β,γ) or a (..., 6) torch tensor and
return the matching torch result.

Differentiable through all six lattice parameters.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    import torch
    Tensor = torch.Tensor
    LatticeLike = Union[Sequence[float], "torch.Tensor"]


_DEG2RAD = math.pi / 180.0


def _as_tensor(params, *, dtype, device):
    import torch
    return torch.as_tensor(params, dtype=dtype, device=device)


def metric_tensor(lattice_params, *, dtype=None, device=None):
    """Direct-space metric tensor G (3, 3).  Differentiable in (a,b,c,α,β,γ)."""
    import torch
    p = _as_tensor(lattice_params, dtype=dtype or torch.float64, device=device)
    a, b, c = p[..., 0], p[..., 1], p[..., 2]
    al, be, ga = p[..., 3] * _DEG2RAD, p[..., 4] * _DEG2RAD, p[..., 5] * _DEG2RAD
    ca, cb, cg = torch.cos(al), torch.cos(be), torch.cos(ga)

    G = torch.stack([
        torch.stack([a * a,        a * b * cg,  a * c * cb], dim=-1),
        torch.stack([a * b * cg,   b * b,       b * c * ca], dim=-1),
        torch.stack([a * c * cb,   b * c * ca,  c * c     ], dim=-1),
    ], dim=-2)
    return G


def reciprocal_metric_tensor(lattice_params, *, dtype=None, device=None):
    """G* = G⁻¹.  Differentiable."""
    import torch
    G = metric_tensor(lattice_params, dtype=dtype, device=device)
    return torch.linalg.inv(G)


def cell_volume(lattice_params, *, dtype=None, device=None):
    import torch
    G = metric_tensor(lattice_params, dtype=dtype, device=device)
    return torch.sqrt(torch.linalg.det(G))


def d_spacing(hkl, lattice_params, *, dtype=None, device=None):
    """d_hkl in Å for one or many reflections.

    ``hkl`` is shape (..., 3) (any int/float type).  Returns shape (...,).
    Differentiable through ``lattice_params``.
    """
    import torch
    Gstar = reciprocal_metric_tensor(lattice_params, dtype=dtype, device=device)
    h = torch.as_tensor(hkl, dtype=Gstar.dtype, device=Gstar.device)
    inv_d2 = torch.einsum('...i,ij,...j->...', h, Gstar, h)
    inv_d2 = torch.clamp(inv_d2, min=1e-30)
    return 1.0 / torch.sqrt(inv_d2)


def s_squared(hkl, lattice_params, *, dtype=None, device=None):
    """s² = (sin θ / λ)² · ... wait — ``s² = 1/(2d)²``  is INDEPENDENT of λ.

    Returns sin²θ/λ² = 1/(4 d²).  Used in Cromer-Mann form factors and DWF.
    Differentiable through lattice_params.
    """
    d = d_spacing(hkl, lattice_params, dtype=dtype, device=device)
    return 1.0 / (4.0 * d * d)


def two_theta(hkl, lattice_params, wavelength_A, *, dtype=None, device=None):
    """2θ in radians.  Differentiable through lattice_params and wavelength."""
    import torch
    d = d_spacing(hkl, lattice_params, dtype=dtype, device=device)
    wl = torch.as_tensor(wavelength_A, dtype=d.dtype, device=d.device)
    sin_th = wl / (2.0 * d)
    sin_th = torch.clamp(sin_th, min=-1.0, max=1.0)
    return 2.0 * torch.asin(sin_th)

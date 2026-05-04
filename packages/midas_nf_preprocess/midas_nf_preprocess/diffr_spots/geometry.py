"""Bragg geometry for the diffr_spots pipeline.

We **delegate** the omega/eta calculation to
``midas_diffract.HEDMForwardModel.calc_bragg_geometry`` (forward.py:680-857)
to avoid maintaining two copies of the same quadratic-Bragg solver. That
function ports ``CalcDiffractionSpots.c::CalcOmega`` (NF) and
``ForwardSimulationCompressed.c`` (FF), and adds a wedge correction.

Only the helpers that are MakeDiffrSpots-specific live here:

  - ``calc_spot_position`` : ``(eta, ring_radius) -> (yl, zl)`` lab-frame
    projection at a given detector distance (matches MakeDiffrSpots.c L115-L120).
  - ``rotate_around_z``    : convenience helper used by tests.

Returned omega/eta from calc_bragg_geometry are in radians; the rest of this
package converts to degrees at the boundary so the binary output matches the
C convention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


def rotate_around_z(v: torch.Tensor, alpha_deg: torch.Tensor) -> torch.Tensor:
    """Apply a rotation about the +Z axis (active, right-handed).

    Direct port of ``RotateAroundZ`` (MakeDiffrSpots.c L99-L107). Used in tests
    to validate the geometry; the production path goes through midas_diffract.
    """
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3, got shape {tuple(v.shape)}")
    a = alpha_deg * _DEG2RAD
    cosa = torch.cos(a)
    sina = torch.sin(a)
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    return torch.stack(
        [cosa * x - sina * y, sina * x + cosa * y, z], dim=-1
    )


def calc_eta_deg(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Convert lab-frame (y, z) into eta in degrees.

    Port of ``MakeDiffrSpots.c::CalcEtaAngle`` (L109-L113), used as a
    cross-check against the radians version returned by midas_diffract.
    """
    r = torch.sqrt(y * y + z * z)
    eps = torch.finfo(r.dtype).eps
    cos_eta = torch.where(r > 0, z / torch.clamp(r, min=eps), torch.zeros_like(z))
    cos_eta = torch.clamp(cos_eta, -1.0, 1.0)
    alpha = torch.acos(cos_eta) * _RAD2DEG
    return torch.where(y > 0, -alpha, alpha)


def calc_spot_position(
    ring_radius: torch.Tensor, eta_deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lab-frame ``(yl, zl)`` from ring radius and eta (degrees).

    Direct port of ``MakeDiffrSpots.c::CalcSpotPosition`` (L115-L120):

        yl = -RingRadius * sin(eta)
        zl =  RingRadius * cos(eta)

    This is the bit MakeDiffrSpots needs but ``midas_diffract`` does not
    expose -- the latter goes straight from eta to detector pixel coordinates,
    skipping the lab-frame intermediate.
    """
    eta_rad = eta_deg * _DEG2RAD
    yl = -ring_radius * torch.sin(eta_rad)
    zl = ring_radius * torch.cos(eta_rad)
    return yl, zl


# -----------------------------------------------------------------------------
# Lazy adapter to midas_diffract.HEDMForwardModel for the Bragg geometry call.
# -----------------------------------------------------------------------------


def _make_minimal_geometry(
    distance_um: float, *, n_pixels: int = 2048, wedge_deg: float = 0.0
):
    """Build an HEDMGeometry whose pixel/beam fields are placeholders.

    ``calc_bragg_geometry`` only reads ``self.wedge`` and ``self.epsilon``
    (the Bragg quadratic itself doesn't touch pixels or beam centers), so any
    consistent values for the unused fields work.
    """
    from midas_diffract.forward import HEDMGeometry

    return HEDMGeometry(
        Lsd=float(distance_um),
        y_BC=0.0,
        z_BC=0.0,
        px=1.0,
        omega_start=0.0,
        omega_step=0.1,
        n_frames=1,
        n_pixels_y=n_pixels,
        n_pixels_z=n_pixels,
        min_eta=0.0,
        wavelength=0.0,
        wedge=float(wedge_deg),
        flip_y=False,
    )


def bragg_omega_eta(
    orient_mats: torch.Tensor,
    hkls: torch.Tensor,
    thetas_deg: torch.Tensor,
    *,
    distance_um: float,
    wedge_deg: float = 0.0,
    device: Optional[torch.device] = None,
):
    """Compute (omega_deg, eta_deg, valid) by delegating to midas_diffract.

    Wraps ``HEDMForwardModel.calc_bragg_geometry``: the work happens there.
    Output is converted from radians (midas_diffract) to degrees (the
    convention used by MakeDiffrSpots and the rest of this package).

    Parameters
    ----------
    orient_mats : Tensor of shape ``(N, 3, 3)``.
    hkls        : Tensor of shape ``(M, 3)``.
    thetas_deg  : Tensor of shape ``(M,)`` -- Bragg angles in degrees.

    Returns
    -------
    omega_deg : Tensor of shape ``(N, 2, M)`` -- two solutions per HKL.
    eta_deg   : Tensor of shape ``(N, 2, M)``.
    valid     : Bool Tensor of shape ``(N, 2, M)``.
    """
    from midas_diffract.forward import HEDMForwardModel

    if orient_mats.ndim != 3 or orient_mats.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected (N, 3, 3), got shape {tuple(orient_mats.shape)}"
        )
    if hkls.shape[-1] != 3:
        raise ValueError(f"Expected hkls last dim = 3, got {tuple(hkls.shape)}")

    device = device or orient_mats.device

    # Construct a one-shot model. The constructor stores hkls/thetas as
    # buffers, but calc_bragg_geometry will use the explicit args we pass.
    geom = _make_minimal_geometry(distance_um, wedge_deg=wedge_deg)
    model = HEDMForwardModel(
        hkls=hkls.to(device),
        thetas=thetas_deg.to(device) * _DEG2RAD,  # midas_diffract expects radians
        geometry=geom,
        device=device,
    )

    omega, eta, _two_theta, valid = model.calc_bragg_geometry(
        orientation_matrices=orient_mats,
        hkls_cart=hkls.to(device),
        thetas=thetas_deg.to(device) * _DEG2RAD,
    )
    # midas_diffract layout: (..., 2N, M) -- the leading 2N is a *concatenation*
    # of the two omega solutions across the N axis. For our (N, 1, 3, 3) input
    # we get omega shape (1, 2*N, M); with N=N orientations this becomes
    # (2*N, M) which we reshape into (N, 2, M).
    # In our use we always pass orient_mats of shape (N, 3, 3) which makes
    # midas_diffract interpret it as N "voxels" in a 1D batch. The concat
    # along the K=2N axis stacks "omega_p (N entries)" then "omega_n (N
    # entries)", so reshape with stride pattern needs care.
    # Output of calc_bragg_geometry for our input shape:
    #   omega: (2*N, M)  [no leading batch dim]
    # The first N rows are omega_p (one per orientation); the next N rows
    # are omega_n (same orientations, the other quadratic root). Reshape to
    # (2, N, M) then permute to (N, 2, M).
    n = orient_mats.shape[0]
    omega_2nm = omega.reshape(2, n, -1).permute(1, 0, 2).contiguous()
    eta_2nm = eta.reshape(2, n, -1).permute(1, 0, 2).contiguous()
    valid_2nm = valid.reshape(2, n, -1).permute(1, 0, 2).bool().contiguous()

    return (
        omega_2nm * _RAD2DEG,
        eta_2nm * _RAD2DEG,
        valid_2nm,
    )

"""Tiny synthetic single-grain fixture used across tests.

Builds a far-field-ish HEDMForwardModel with a small reflection list, a
ground-truth grain state, the resulting noise-free spot list, and the
"observed" spot view that the refiner will consume. The fixture is small
enough that the residual layer is exercised end-to-end in <1 s per call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, HEDMGeometry  # type: ignore

DEG2RAD = math.pi / 180.0


@dataclass
class SyntheticFixture:
    model: HEDMForwardModel
    pred_ring_slot: torch.Tensor             # (M,) — ring slot per reflection
    ring_numbers: list[int]
    init_lattice: torch.Tensor               # (6,)
    px: float
    y_BC: float
    z_BC: float

    gt_position: torch.Tensor                # (3,) um
    gt_euler: torch.Tensor                   # (3,) rad
    gt_lattice: torch.Tensor                 # (6,)

    obs_yz_um: torch.Tensor                  # (S, 2) wedge+det-corr lab um
    obs_omega: torch.Tensor                  # (S,) rad
    obs_eta: torch.Tensor                    # (S,) rad
    obs_two_theta: torch.Tensor              # (S,) rad
    obs_ring: torch.Tensor                   # (S,) int — ring slot index
    obs_spot_id: torch.Tensor                # (S,) int

    # GT matching: each observed spot's (K, M) slot in the predicted grid.
    gt_k_idx: torch.Tensor                   # (S,) int
    gt_m_idx: torch.Tensor                   # (S,) int


def make_synthetic(
    *, device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> SyntheticFixture:
    # Cubic reference. Build a richer reflection list (hkls up to |h|² ≤ 6)
    # so the orientation is well-determined and the optimizer's Hessian is
    # not dominated by 1-2 axes. Real FF-HEDM datasets carry far more
    # reflections than this; the test fixture only needs to be large enough
    # that all three Euler angles have comparable gradient magnitudes.
    raw = []
    for h in range(-2, 3):
        for k in range(-2, 3):
            for l in range(-2, 3):
                s2 = h * h + k * k + l * l
                if 0 < s2 <= 6:
                    raw.append([h, k, l])
    hkls_int = torch.tensor(raw, dtype=torch.float64)
    a = 4.04
    wavelength = 0.1729  # Å (24 keV-ish)

    # Reciprocal-lattice basis for cubic a=4.04 Å.
    a_star = 1.0 / a
    hkls_cart = hkls_int.clone() * a_star  # 1/Å, cubic
    G = hkls_cart.norm(dim=-1)              # 1/Å
    sin_th = wavelength * G / 2.0
    sin_th = sin_th.clamp(-0.999, 0.999)
    thetas = torch.asin(sin_th)             # rad

    # FF-mode geometry.
    px = 200.0  # um/pixel
    geom = HEDMGeometry(
        Lsd=1_000_000.0,
        y_BC=1024.0,
        z_BC=1024.0,
        px=px,
        omega_start=-180.0,
        omega_step=0.25,
        n_frames=1440,
        n_pixels_y=2048,
        n_pixels_z=2048,
        min_eta=6.0,
        wavelength=wavelength,
        flip_y=True,
    )
    model = HEDMForwardModel(
        hkls_cart, thetas, geom,
        hkls_int=hkls_int,
        device=device,
    )
    # Cast the model's buffers to the requested dtype.
    model = model.to(dtype=dtype if dtype.is_floating_point else None)

    # Group reflections by |h|² + |k|² + |l|² → ring slot.
    h2 = hkls_int.long().pow(2).sum(dim=-1)
    unique_h2, ring_slots = torch.unique(h2, sorted=True, return_inverse=True)
    pred_ring_slot = ring_slots.to(device=device)
    ring_numbers = list(range(int(unique_h2.numel())))  # 0-based ring indices

    # Ground-truth grain — picked to break the phi1/phi2 quasi-degeneracy
    # that arises for small Phi in cubic systems (otherwise the fit can drift
    # along an near-flat ridge of the loss landscape).
    gt_position = torch.tensor([10.0, -5.0, 2.0], dtype=dtype, device=device)
    gt_euler = torch.tensor(
        [25.0 * DEG2RAD, 38.0 * DEG2RAD, -47.0 * DEG2RAD],
        dtype=dtype, device=device,
    )
    gt_lattice = torch.tensor(
        [a, a, a, 90.0, 90.0, 90.0],
        dtype=dtype, device=device,
    )

    spots = model(gt_euler.view(1, 1, 3), gt_position.view(1, 1, 3),
                  lattice_params=gt_lattice.view(1, 6))

    # Squeeze (B=1, N=1) leading dims off (..., K=2, M).
    def _sq(t):
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
            if t.dim() == 0:
                break
        return t

    omega = _sq(spots.omega)
    eta = _sq(spots.eta)
    two_theta = _sq(spots.two_theta)
    y_pixel = _sq(spots.y_pixel)
    z_pixel = _sq(spots.z_pixel)
    valid = _sq(spots.valid).bool()
    M = omega.shape[-1]

    # Flatten (K, M) -> (K*M,) and keep only valid entries.
    flat_idx = torch.nonzero(valid.reshape(-1)).squeeze(-1)
    if flat_idx.numel() == 0:
        raise RuntimeError("synthetic fixture: no valid spots")

    k_idx = flat_idx // M
    m_idx = flat_idx %  M

    obs_yz_um = torch.stack([
        (geom.y_BC - y_pixel.reshape(-1)[flat_idx]) * px,
        (z_pixel.reshape(-1)[flat_idx] - geom.z_BC) * px,
    ], dim=-1)
    obs_omega = omega.reshape(-1)[flat_idx]
    obs_eta = eta.reshape(-1)[flat_idx]
    obs_two_theta = two_theta.reshape(-1)[flat_idx]
    obs_ring = pred_ring_slot[m_idx]
    obs_spot_id = torch.arange(flat_idx.numel(), dtype=torch.int64, device=device)

    return SyntheticFixture(
        model=model,
        pred_ring_slot=pred_ring_slot,
        ring_numbers=ring_numbers,
        init_lattice=gt_lattice.clone(),
        px=px, y_BC=geom.y_BC, z_BC=geom.z_BC,
        gt_position=gt_position,
        gt_euler=gt_euler,
        gt_lattice=gt_lattice,
        obs_yz_um=obs_yz_um,
        obs_omega=obs_omega,
        obs_eta=obs_eta,
        obs_two_theta=obs_two_theta,
        obs_ring=obs_ring,
        obs_spot_id=obs_spot_id,
        gt_k_idx=k_idx,
        gt_m_idx=m_idx,
    )


def gt_match(fix: SyntheticFixture, *, device: torch.device,
             dtype: torch.dtype):
    """Return a MatchResult with every observed spot mapped to its GT slot."""
    from midas_fit_grain.matching import MatchResult
    S = fix.gt_k_idx.shape[0]
    return MatchResult(
        k_idx=fix.gt_k_idx.to(device=device),
        m_idx=fix.gt_m_idx.to(device=device),
        mask=torch.ones(S, dtype=torch.bool, device=device),
        delta_omega=torch.zeros(S, dtype=dtype, device=device),
        delta_eta=torch.zeros(S, dtype=dtype, device=device),
    )


def fixture_to_observed(fix: SyntheticFixture, *,
                        device: torch.device, dtype: torch.dtype):
    """Convert SyntheticFixture's spot list into an ObservedSpots."""
    from midas_fit_grain.observations import ObservedSpots
    S = fix.obs_yz_um.shape[0]
    zeros = torch.zeros(S, dtype=dtype, device=device)
    return ObservedSpots(
        spot_id=fix.obs_spot_id.to(device=device),
        ring_nr=fix.obs_ring.to(device=device),     # already a slot index
        y_lab=fix.obs_yz_um[:, 0].to(dtype=dtype),
        z_lab=fix.obs_yz_um[:, 1].to(dtype=dtype),
        omega=fix.obs_omega.to(dtype=dtype),
        eta=fix.obs_eta.to(dtype=dtype),
        two_theta=fix.obs_two_theta.to(dtype=dtype),
        grain_radius=zeros.clone(),
        fit_rmse=zeros.clone(),
        y_orig=zeros.clone(),
        z_orig=zeros.clone(),
        omega_ini=fix.obs_omega.to(dtype=dtype),
        mask_touched=zeros.clone(),
    )

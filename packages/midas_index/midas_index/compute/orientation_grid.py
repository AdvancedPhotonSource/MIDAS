"""Orientation-grid construction.

Mirrors `GenerateCandidateOrientationsF` from
`FF_HEDM/src/IndexerOMP.c:674`.

Algorithm:
  1. Pre-rotation R1: align `hkl` with `plane_normal` via axis = hkl x normal,
     angle = acos(dot/|hkl||normal|).
  2. Sweep R2(angle) around `plane_normal` for angle in [0, MaxAngle) steps of
     `stepsize_orient_deg`.
  3. Each candidate orientation is `R3 = R2 @ R1`.

`MaxAngle` is symmetry-scaled via `calc_rotation_angle`.
"""

from __future__ import annotations

import math

import torch

from .rotation import axis_angle_batch, calc_rotation_angle


def generate_candidate_orientations(
    hkl: torch.Tensor,
    plane_normal: torch.Tensor,
    stepsize_orient_deg: float,
    *,
    ring_nr: int,
    space_group: int,
    hkl_int: tuple[int, int, int],
    abcabg: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    """Build the orientation matrices for one (hkl, plane_normal) seed.

    Parameters
    ----------
    hkl : torch.Tensor (3,)
        The hkl Cartesian G-vector for this seed reflection (RingHKL row).
    plane_normal : torch.Tensor (3,)
        Unit normal direction in the lab frame (from the seed spot's geometry).
    stepsize_orient_deg : float
        Angular step in degrees (`IndexerParams.StepsizeOrient`).
    ring_nr, space_group, hkl_int, abcabg
        Used for `calc_rotation_angle`.

    Returns
    -------
    or_mats : torch.Tensor (n_or, 3, 3) on the input's device/dtype.
        n_or = floor(MaxAngle / stepsize_orient_deg). May be 0 if MaxAngle == 0.
    """
    device = hkl.device
    dtype = hkl.dtype

    # Pre-rotation: axis = hkl x normal, angle = acos(dot / |hkl||normal|)
    v = torch.linalg.cross(hkl, plane_normal)
    hkl_len = torch.linalg.vector_norm(hkl)
    pn_len = torch.linalg.vector_norm(plane_normal)
    cos_pre = (torch.dot(hkl, plane_normal) / (hkl_len * pn_len)).clamp(-1.0, 1.0)
    pre_angle_deg = torch.rad2deg(torch.acos(cos_pre))
    R_pre = axis_angle_batch(v, pre_angle_deg)            # (3, 3)

    max_angle_deg = calc_rotation_angle(ring_nr, space_group, hkl_int, abcabg)
    if max_angle_deg <= 0:
        return torch.empty((0, 3, 3), device=device, dtype=dtype)
    n_steps = int(max_angle_deg / stepsize_orient_deg)
    if n_steps == 0:
        return torch.empty((0, 3, 3), device=device, dtype=dtype)

    angles = torch.arange(n_steps, device=device, dtype=dtype) * stepsize_orient_deg
    axes = plane_normal.unsqueeze(0).expand(n_steps, 3)
    R_sweep = axis_angle_batch(axes, angles)              # (n_steps, 3, 3)
    return R_sweep @ R_pre                                # (n_steps, 3, 3)


def generate_candidate_orientations_batched(
    hkl: torch.Tensor,                       # (3,)
    plane_normals: torch.Tensor,             # (B, 3)
    stepsize_orient_deg: float,
    *,
    ring_nr: int,
    space_group: int,
    hkl_int: tuple[int, int, int],
    abcabg: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    """Vectorized variant: build orientation grid for B (y0, z0) candidates at once.

    All B candidates share the same hkl/ring/space_group, so `n_steps` is shared.
    Returns shape `(B, n_steps, 3, 3)`. If MaxAngle == 0, returns (B, 0, 3, 3).

    Eliminates B small kernel launches that the scalar variant pays inside the
    per-(y0, z0) inner loop in `_process_seed_group`.
    """
    device = hkl.device
    dtype = hkl.dtype
    B = plane_normals.shape[0]

    max_angle_deg = calc_rotation_angle(ring_nr, space_group, hkl_int, abcabg)
    if max_angle_deg <= 0:
        return torch.empty((B, 0, 3, 3), device=device, dtype=dtype)
    n_steps = int(max_angle_deg / stepsize_orient_deg)
    if n_steps == 0:
        return torch.empty((B, 0, 3, 3), device=device, dtype=dtype)

    # Pre-rotation per (y0, z0) candidate: (B, 3, 3)
    hkl_b = hkl.unsqueeze(0).expand(B, 3)
    v = torch.linalg.cross(hkl_b, plane_normals, dim=-1)        # (B, 3)
    hkl_len = torch.linalg.vector_norm(hkl)                     # scalar
    pn_len = torch.linalg.vector_norm(plane_normals, dim=-1)    # (B,)
    cos_pre = (
        (plane_normals @ hkl) / (hkl_len * pn_len.clamp_min(1e-30))
    ).clamp(-1.0, 1.0)
    pre_angle_deg = torch.rad2deg(torch.acos(cos_pre))           # (B,)
    R_pre = axis_angle_batch(v, pre_angle_deg)                   # (B, 3, 3)

    # Sweep: angles (n_steps,), axes (B, n_steps, 3), angles_b (B, n_steps)
    angles = torch.arange(n_steps, device=device, dtype=dtype) * stepsize_orient_deg
    axes = plane_normals.unsqueeze(1).expand(B, n_steps, 3)
    angles_b = angles.unsqueeze(0).expand(B, n_steps)
    R_sweep = axis_angle_batch(axes, angles_b)                   # (B, n_steps, 3, 3)

    # R_sweep[b, s] @ R_pre[b] for each b, s.
    return R_sweep @ R_pre.unsqueeze(1)                          # (B, n_steps, 3, 3)

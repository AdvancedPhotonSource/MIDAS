"""Rotation utilities — thin shims over `midas-stress.orientation`.

The bulk of orientation math (axis-angle -> R, Euler <-> R, quaternions,
symmetry, FZ reduction) lives in `midas_stress.orientation`. After
midas-stress 0.5.0 (see dev/implementation_plan.md §17), those calls accept
torch tensors and return them on the active device.

The only indexer-specific item that stays here is `calc_rotation_angle`.
"""

from __future__ import annotations

import torch

from midas_stress.orientation import axis_angle_to_orient_mat


# ---------------------------------------------------------------------------
# Symmetry-scaled angular step for the Rodrigues orientation grid
# (mirrors `CalcRotationAngle` from FF_HEDM/src/IndexerOMP.c:597).
# ---------------------------------------------------------------------------


def calc_rotation_angle(
    ring_nr: int,
    space_group: int,
    hkl_int: tuple[int, int, int],
    abcabg: tuple[float, float, float, float, float, float] | None = None,
) -> float:
    """Return the maximum sweep angle (in degrees) for the orientation grid.

    Parameters
    ----------
    ring_nr : int
        Ring index in `IndexerParams.RingNumbers`. Used in the C code for an
        HKLints[][3] lookup; here we accept the resolved (h, k, l) directly
        in `hkl_int` so callers don't have to maintain the global table.
    space_group : int
        Crystal space group (1-230).
    hkl_int : tuple[int, int, int]
        Integer Miller indices for the seed reflection.
    abcabg : optional 6-tuple
        Lattice parameters (a, b, c, alpha, beta, gamma). Required for the
        monoclinic special-case branch (SGNum 3..15); ignored otherwise.

    Returns
    -------
    angle_deg : float
        Maximum angle in degrees. The orientation grid runs [0, angle_deg)
        in steps of `IndexerParams.StepsizeOrient`.
    """
    h, k, l = (abs(int(hkl_int[0])), abs(int(hkl_int[1])), abs(int(hkl_int[2])))
    n_zeros = (h == 0) + (k == 0) + (l == 0)
    if n_zeros == 3:
        return 0.0
    sg = int(space_group)
    if sg in (1, 2):                                    # Triclinic
        return 360.0
    if 3 <= sg <= 15:                                   # Monoclinic
        if n_zeros != 2:
            return 360.0
        if abcabg is None:
            return 360.0
        alpha, _, gamma = abcabg[3], abcabg[4], abcabg[5]
        if alpha == 90 and abcabg[4] == 90 and l != 0:
            return 180.0
        if alpha == 90 and gamma == 90 and h != 0:
            return 180.0
        if alpha == 90 and gamma == 90 and k != 0:
            return 180.0
        return 360.0
    if 16 <= sg <= 74:                                  # Orthorhombic
        return 180.0 if n_zeros == 2 else 360.0
    if 75 <= sg <= 142:                                 # Tetragonal
        if n_zeros == 0:
            return 360.0
        if n_zeros == 1 and l == 0 and h == k:
            return 180.0
        if n_zeros == 2:
            return 180.0 if l == 0 else 90.0
        return 360.0
    if 143 <= sg <= 167:                                # Trigonal
        if n_zeros == 2 and l != 0:
            return 120.0
        return 360.0
    if 168 <= sg <= 194:                                # Hexagonal
        if n_zeros == 2 and l != 0:
            return 60.0
        return 360.0
    if 195 <= sg <= 230:                                # Cubic
        if n_zeros == 2:
            return 90.0
        if n_zeros == 1 and (h == k or k == l or h == l):
            return 180.0
        if h == k and k == l:
            return 120.0
        return 360.0
    return 0.0


# ---------------------------------------------------------------------------
# Axis-angle -> rotation matrix (batched, torch-native via midas-stress)
# ---------------------------------------------------------------------------


def axis_angle_batch(
    axes: torch.Tensor,
    angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Stacked axis-angle -> rotation matrices.

    Parameters
    ----------
    axes : torch.Tensor, shape (..., 3)
    angles_deg : torch.Tensor, shape broadcastable to leading dims of `axes`

    Returns
    -------
    R : torch.Tensor, shape (..., 3, 3) on the same device/dtype as `axes`.
    """
    return axis_angle_to_orient_mat(axes, angles_deg)

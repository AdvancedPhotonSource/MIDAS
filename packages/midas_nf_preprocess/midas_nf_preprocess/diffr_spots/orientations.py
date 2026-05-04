"""Orientation conversions: quaternions <-> rotation matrices, Euler angles.

Direct vectorized PyTorch ports of:

  - ``MakeDiffrSpots.c::QuatToOrientMat`` (L77-L97) for the
    quaternion-to-matrix path used by MakeDiffrSpots.
  - Bunge ZXZ Euler angles to rotation matrix (used by FF seeds).

All functions are batched over the leading axis.
"""

from __future__ import annotations

import math

import torch


def quat_to_orient_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Direct port of ``MakeDiffrSpots.c::QuatToOrientMat`` (L77-L97). Quaternion
    convention: ``(w, x, y, z)`` with ``w`` first (matching the C ``Quat[0]``).

    Parameters
    ----------
    quats : Tensor of shape ``(..., 4)``.

    Returns
    -------
    Tensor of shape ``(..., 3, 3)``.
    """
    if quats.shape[-1] != 4:
        raise ValueError(
            f"Expected last dim = 4 (w, x, y, z), got shape {tuple(quats.shape)}"
        )
    w, x, y, z = quats.unbind(-1)
    # Match C names: Q1 = x, Q2 = y, Q3 = z; Q0 = w.
    Q1_2 = x * x
    Q2_2 = y * y
    Q3_2 = z * z
    Q12 = x * y
    Q03 = w * z
    Q13 = x * z
    Q02 = w * y
    Q23 = y * z
    Q01 = w * x

    one = torch.ones_like(Q1_2)
    two = 2.0
    m00 = one - two * (Q2_2 + Q3_2)
    m01 = two * (Q12 - Q03)
    m02 = two * (Q13 + Q02)
    m10 = two * (Q12 + Q03)
    m11 = one - two * (Q1_2 + Q3_2)
    m12 = two * (Q23 - Q01)
    m20 = two * (Q13 - Q02)
    m21 = two * (Q23 + Q01)
    m22 = one - two * (Q1_2 + Q2_2)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def orient_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quat_to_orient_matrix`. Returns quaternion ``(w, x, y, z)``.

    Uses Shepperd's method (numerically stable: pick the largest of
    ``trace`` and the diagonal entries to avoid divide-by-near-zero).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got shape {tuple(R.shape)}")
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    # Four candidate q components (case-by-case)
    eps = torch.finfo(R.dtype).eps
    # We compute all four and pick the safest per element.
    t0 = trace + 1.0
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22
    cases = torch.stack([t0, t1, t2, t3], dim=-1)
    best = cases.argmax(dim=-1)

    s0 = 2.0 * torch.sqrt(torch.clamp(t0, min=eps))
    s1 = 2.0 * torch.sqrt(torch.clamp(t1, min=eps))
    s2 = 2.0 * torch.sqrt(torch.clamp(t2, min=eps))
    s3 = 2.0 * torch.sqrt(torch.clamp(t3, min=eps))

    # Case 0: trace dominant
    w0 = 0.25 * s0
    x0 = (R[..., 2, 1] - R[..., 1, 2]) / s0
    y0 = (R[..., 0, 2] - R[..., 2, 0]) / s0
    z0 = (R[..., 1, 0] - R[..., 0, 1]) / s0

    # Case 1: m00 dominant
    w1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    x1 = 0.25 * s1
    y1 = (R[..., 0, 1] + R[..., 1, 0]) / s1
    z1 = (R[..., 0, 2] + R[..., 2, 0]) / s1

    # Case 2: m11 dominant
    w2 = (R[..., 0, 2] - R[..., 2, 0]) / s2
    x2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
    y2 = 0.25 * s2
    z2 = (R[..., 1, 2] + R[..., 2, 1]) / s2

    # Case 3: m22 dominant
    w3 = (R[..., 1, 0] - R[..., 0, 1]) / s3
    x3 = (R[..., 0, 2] + R[..., 2, 0]) / s3
    y3 = (R[..., 1, 2] + R[..., 2, 1]) / s3
    z3 = 0.25 * s3

    candidates = torch.stack(
        [
            torch.stack([w0, x0, y0, z0], dim=-1),
            torch.stack([w1, x1, y1, z1], dim=-1),
            torch.stack([w2, x2, y2, z2], dim=-1),
            torch.stack([w3, x3, y3, z3], dim=-1),
        ],
        dim=-2,
    )
    chosen = torch.gather(
        candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)
    ).squeeze(-2)
    # Canonicalize: enforce non-negative w.
    sign = torch.where(chosen[..., 0] < 0, -torch.ones_like(chosen[..., 0]), torch.ones_like(chosen[..., 0]))
    return chosen * sign[..., None]


def euler_to_orient_matrix(
    euler_deg: torch.Tensor, *, convention: str = "ZXZ"
) -> torch.Tensor:
    """Convert Bunge Euler angles (deg) to rotation matrices.

    Default convention is Bunge ZXZ (``phi1, Phi, phi2``), the MIDAS standard.

    Parameters
    ----------
    euler_deg : Tensor of shape ``(..., 3)``, in degrees.
    convention : "ZXZ" (Bunge, default).

    Returns
    -------
    Tensor of shape ``(..., 3, 3)``.
    """
    if convention.upper() != "ZXZ":
        raise NotImplementedError(f"Only Bunge ZXZ is supported; got {convention!r}")
    if euler_deg.shape[-1] != 3:
        raise ValueError(
            f"Expected last dim = 3 (phi1, Phi, phi2), got shape {tuple(euler_deg.shape)}"
        )
    rad = euler_deg * (math.pi / 180.0)
    phi1, Phi, phi2 = rad.unbind(-1)
    c1, s1 = torch.cos(phi1), torch.sin(phi1)
    cP, sP = torch.cos(Phi), torch.sin(Phi)
    c2, s2 = torch.cos(phi2), torch.sin(phi2)
    # Bunge passive rotation matrix.
    m00 = c1 * c2 - s1 * s2 * cP
    m01 = s1 * c2 + c1 * s2 * cP
    m02 = s2 * sP
    m10 = -c1 * s2 - s1 * c2 * cP
    m11 = -s1 * s2 + c1 * c2 * cP
    m12 = c2 * sP
    m20 = s1 * sP
    m21 = -c1 * sP
    m22 = cP
    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

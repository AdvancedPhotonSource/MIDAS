"""Cluster-level symmetry alignment.

Given a cluster's representative orientation ``q_rep`` and a member orientation
``q_member`` (both as unit quaternions in the (w, x, y, z) layout used by
``midas_stress``), pick the symmetry op that brings ``q_member`` into the
variant of ``q_rep`` with the smallest residual angle.

This is the key step that makes the spot-aware merge symmetry-aware: the row
permutation of the IndexBestFull / FitBest tables must be applied with this
op so that "row k" means the same physical reflection across all members of
the cluster.

Convention (matches ``midas_stress.orientation``):

  - quaternion layout: (w, x, y, z), real part first
  - ``q · r``: Hamilton product
  - misorientation = 2 · arccos(|w(q1^{-1} · q2 · S)|) for some S in the
    point group; we minimise over S

Returns the **index** of the chosen op (so the caller can look up
``hkl_perm[idx]`` from a precomputed :class:`SymmetryTable`).
"""

from __future__ import annotations

from typing import Tuple

import torch


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of unit quaternion (real-first layout)."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, real-first layout. Broadcasts on leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def pick_best_sym_op(
    q_rep: torch.Tensor,
    q_member: torch.Tensor,
    sym_quats: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    """Return ``(s_idx, residual_angle_rad)`` minimising residual misorientation.

    Parameters
    ----------
    q_rep : torch.Tensor
        ``(4,)`` rep quaternion (w, x, y, z), unit norm.
    q_member : torch.Tensor
        ``(4,)`` member quaternion.
    sym_quats : torch.Tensor
        ``(n_sym, 4)`` from :class:`SymmetryTable.ops_quat`.

    Returns
    -------
    s_idx : int
        Index in ``sym_quats`` of the op that minimises residual angle.
    residual_angle : torch.Tensor
        Scalar tensor: ``2 · arccos(|w(q_rep^{-1} · q_member · S)|)`` at the
        best ``S``.
    """
    if q_rep.shape != (4,) or q_member.shape != (4,):
        raise ValueError("q_rep and q_member must each be shape (4,)")
    if sym_quats.dim() != 2 or sym_quats.shape[1] != 4:
        raise ValueError(f"sym_quats must be (n_sym, 4); got {sym_quats.shape}")

    # Compute, for every S: q_member · S, then q_rep^{-1} · (q_member · S),
    # then |w| → smaller |w| means larger angle, so we want max |w|.
    # We broadcast q_member over the n_sym leading dimension of sym_quats.
    qm_expand = q_member.unsqueeze(0).expand_as(sym_quats)             # (n_sym, 4)
    qm_S = _quat_mul(qm_expand, sym_quats)                              # (n_sym, 4)
    qrep_inv = _quat_inv(q_rep).unsqueeze(0).expand_as(qm_S)
    delta = _quat_mul(qrep_inv, qm_S)                                   # (n_sym, 4)
    w_abs = delta[..., 0].abs()
    s_idx = int(torch.argmax(w_abs).item())
    angle = 2.0 * torch.arccos(torch.clamp(w_abs[s_idx], -1.0, 1.0))
    return s_idx, angle


def align_member_to_rep(
    member_table: torch.Tensor,
    s_idx: int,
    hkl_perm: torch.Tensor,
) -> torch.Tensor:
    """Permute a per-row table from member's variant frame to the rep's frame.

    Parameters
    ----------
    member_table : torch.Tensor
        ``(n_hkls, ...)``. Typically the member's slice of ``IndexBestFull``
        (shape ``(n_hkls, 2)``) or the per-spot row from ``FitBest``
        (shape ``(n_hkls, 22)``).
    s_idx : int
        Index of the symmetry op chosen by :func:`pick_best_sym_op`.
    hkl_perm : torch.Tensor
        ``(n_sym, n_hkls)`` int64. From :attr:`SymmetryTable.hkl_perm`.

    Returns
    -------
    aligned : torch.Tensor
        Same shape as ``member_table``; ``aligned[k] = member_table[π_s[k]]``
        where ``π_s = hkl_perm[s_idx]``. Rows where ``π_s[k] == -1`` are
        zeroed (so they neither vote nor count toward agreement metrics).
    """
    perm = hkl_perm[s_idx]                                               # (n_hkls,)
    valid = perm >= 0
    safe = torch.where(valid, perm, torch.zeros_like(perm))
    out = member_table[safe]
    if not bool(valid.all()):
        out = out.clone()
        # broadcast valid into all trailing dims
        zero_mask = ~valid
        # member_table.shape[1:] is the trailing dims
        view_shape = (perm.shape[0],) + (1,) * (member_table.dim() - 1)
        out = torch.where(
            zero_mask.view(view_shape).expand_as(out),
            torch.zeros_like(out),
            out,
        )
    return out

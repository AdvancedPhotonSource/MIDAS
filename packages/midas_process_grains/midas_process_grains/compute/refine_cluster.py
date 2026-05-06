"""Phase 2 — symmetry-aware spot-aware sub-clustering.

Operates inside each Phase-1 candidate group. For every member of the group:

  1. Pick the cluster representative (min-IA).
  2. Symmetry-align every other member's hkl-row table to the rep's variant
     frame using :func:`canonicalize.pick_best_sym_op` and
     :func:`canonicalize.align_member_to_rep`.
  3. Build a seed-pair graph with edge weights:

         w_ab = α · A_ab + (1 − α) · J_ab

     where  ``J_ab`` = Jaccard of the SpotID sets (block A; symmetry-invariant)
     and    ``A_ab`` = per-hkl SpotID-agreement at "informative" hkls only
                       (block B + block C).
  4. Drop edges with ``w_ab < τ_w`` and run connected components on what's
     left. Each connected component is a single physical grain.

The "informative" hkls (block C) are those where the pair's residual
misorientation, projected to ring radius, exceeds ``τ_pixel`` detector pixels
of predicted spot displacement; below that, two equivalent-orientation
sub-grains predict the same observed peak by construction, so per-hkl
agreement at that row carries no information and we don't count it.

Outputs (per Phase-1 group):

    grains : list[GrainCandidate]
        each with member positions, rep position, aligned IndexBestFull rows,
        chosen symmetry op per member, and the connected-component label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .canonicalize import align_member_to_rep, pick_best_sym_op
from .symmetry import SymmetryTable


__all__ = [
    "GrainCandidate",
    "refine_cluster_spot_aware",
]


@dataclass
class GrainCandidate:
    """One sub-cluster after Phase 2.

    Attributes
    ----------
    rep_pos : int
        Row index in OrientPosFit.bin (== seed position) of the rep.
    member_positions : list[int]
        All members of this sub-cluster (rep included).
    member_sym_ops : list[int]
        For each member, the index in ``SymmetryTable.ops_quat`` of the op
        that aligns it to the rep. ``0`` is the identity.
    aligned_spot_tables : list[np.ndarray]
        For each member, its ``IndexBestFull[pos, :, 0]`` after applying
        the symmetry-permutation. Same length as ``member_positions``.
    member_misori_rad : np.ndarray
        Per-member residual misorientation (radians) after the chosen op.
    """

    rep_pos: int
    member_positions: List[int]
    member_sym_ops: List[int]
    aligned_spot_tables: List[np.ndarray]
    member_misori_rad: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring_radii_per_hkl(
    hkl_table_real: np.ndarray,
) -> np.ndarray:
    """Pull ring_radius (column 6 of ``HklTable.real``) into a flat (n_hkls,) array."""
    return np.asarray(hkl_table_real[:, 6], dtype=np.float64)


def _spot_set(table_row0: np.ndarray) -> set:
    """Distinct non-zero SpotIDs from a (n_hkls,) col-0 view."""
    arr = table_row0.astype(np.int64, copy=False)
    return set(int(v) for v in arr[arr != 0])


def _per_hkl_informative_mask(
    misori_rad: float,
    ring_radii: np.ndarray,
    pixel_size_um: float,
    pixel_tol: float,
) -> np.ndarray:
    """Block C: row is informative iff predicted spot displacement
    ``misori * ring_radius`` exceeds ``pixel_tol`` detector pixels.

    Both ``misori`` and ``ring_radii`` are in consistent units; multiplying
    misori (rad) × ring_radius (µm) → arc length in µm, divided by pixel size
    gives the predicted shift in pixels.
    """
    shift_um = abs(misori_rad) * ring_radii
    return shift_um > (pixel_tol * pixel_size_um)


def _pairwise_weights(
    aligned_tables: List[np.ndarray],
    misori_pair: np.ndarray,
    *,
    ring_radii: np.ndarray,
    pixel_size_um: float,
    pixel_tol: float,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pair edge weights inside one Phase-1 cluster.

    Parameters
    ----------
    aligned_tables : list of np.ndarray
        Each ``(n_hkls,)`` col-0 view (matched SpotID per row), already
        symmetry-aligned to the rep's variant frame.
    misori_pair : np.ndarray
        ``(n_members, n_members)`` symmetric, residual misorientation
        between every pair in radians.
    ring_radii, pixel_size_um, pixel_tol : float / array
        For block C — the resolution-aware informative-hkl filter.
    alpha : float
        Mixing coefficient: ``w = alpha * agreement + (1 - alpha) * jaccard``.

    Returns
    -------
    jaccard : (n, n) float64
    agreement : (n, n) float64
    weights : (n, n) float64
    """
    n = len(aligned_tables)
    jaccard = np.eye(n, dtype=np.float64)
    agreement = np.eye(n, dtype=np.float64)
    weights = np.eye(n, dtype=np.float64)

    spot_sets = [_spot_set(t) for t in aligned_tables]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = spot_sets[i], spot_sets[j]
            union = si | sj
            inter = si & sj
            J = (len(inter) / len(union)) if union else 0.0
            jaccard[i, j] = jaccard[j, i] = J

            informative = _per_hkl_informative_mask(
                misori_pair[i, j], ring_radii, pixel_size_um, pixel_tol,
            )
            ti = aligned_tables[i].astype(np.int64, copy=False)
            tj = aligned_tables[j].astype(np.int64, copy=False)
            both_claim = (ti != 0) & (tj != 0) & informative
            if both_claim.any():
                agree_count = int(((ti == tj) & both_claim).sum())
                A = agree_count / int(both_claim.sum())
            else:
                # Below resolution / no overlap with informative rows:
                # fall back to Jaccard.
                A = J
            agreement[i, j] = agreement[j, i] = A

            weights[i, j] = weights[j, i] = alpha * A + (1.0 - alpha) * J
    return jaccard, agreement, weights


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def refine_cluster_spot_aware(
    member_positions: Sequence[int],
    rep_pos: int,
    member_quats: np.ndarray,
    rep_quat: np.ndarray,
    member_index_best_full_col0: np.ndarray,
    sym_table: SymmetryTable,
    *,
    ring_radii_per_hkl: np.ndarray,
    pixel_size_um: float,
    pixel_tol: float = 1.0,
    jaccard_tol: float = 0.5,
    agreement_tol: float = 0.7,
    merge_alpha: float = 0.6,
    edge_weight_threshold: float = 0.7,
    min_nr_spots: int = 2,
) -> List[GrainCandidate]:
    """Run Phase 2 inside one Phase-1 candidate group.

    Parameters
    ----------
    member_positions : sequence[int]
        Row indices in OrientPosFit.bin of all seeds in the group, including
        the rep at ``rep_pos`` (must be present in ``member_positions``).
    rep_pos : int
        The chosen rep's seed-row index.
    member_quats : np.ndarray
        ``(n_members, 4)`` quaternions, in the same order as ``member_positions``.
    rep_quat : np.ndarray
        ``(4,)`` rep quaternion.
    member_index_best_full_col0 : np.ndarray
        ``(n_members, n_hkls)`` int64. SpotID column of IndexBestFull for
        each member, in member order.
    sym_table : SymmetryTable
        From :func:`build_symmetry_table`.
    ring_radii_per_hkl : np.ndarray
        ``(n_hkls,)`` ring radius (µm) for each theoretical hkl row.
    pixel_size_um : float
        Detector pixel pitch.
    pixel_tol : float
        Block-C informative-hkl threshold (in pixels).
    jaccard_tol, agreement_tol : float
        Floors below which an edge is considered weak; used as guardrails for
        diagnostic logging only — the actual cut is on ``edge_weight_threshold``.
    merge_alpha : float
        Weight blend coefficient (Eq. in module docstring).
    edge_weight_threshold : float
        Cut for the connected-component graph.
    min_nr_spots : int
        Floor on grain spot count. Sub-clusters with fewer matched-spot
        union entries are dropped here. (Cluster size — number of seeds —
        is enforced upstream by Phase 1's cluster size; this is the spot-side
        floor.)

    Returns
    -------
    grains : list[GrainCandidate]
    """
    n = len(member_positions)
    if member_quats.shape[0] != n or member_index_best_full_col0.shape[0] != n:
        raise ValueError(
            f"member arrays must agree on length {n}: "
            f"member_quats {member_quats.shape}, "
            f"col0 {member_index_best_full_col0.shape}"
        )
    if rep_pos not in member_positions:
        raise ValueError(
            f"rep_pos {rep_pos} must be one of member_positions {list(member_positions)}"
        )

    # 1. Symmetry-align each member to the rep's variant frame.
    rep_q_t = torch.from_numpy(rep_quat).to(sym_table.ops_quat)
    sym_quats = sym_table.ops_quat
    member_sym_ops = [0] * n
    member_misori = np.zeros(n, dtype=np.float64)
    aligned_tables: List[np.ndarray] = []

    for i in range(n):
        mq = torch.from_numpy(member_quats[i]).to(sym_table.ops_quat)
        s_idx, residual = pick_best_sym_op(rep_q_t, mq, sym_quats)
        member_sym_ops[i] = int(s_idx)
        member_misori[i] = float(residual)
        # Align this member's col-0 row table.
        member_table = torch.from_numpy(
            member_index_best_full_col0[i].astype(np.float64)
        ).unsqueeze(-1)            # (n_hkls, 1) so align_member_to_rep works on a 2D array
        aligned = align_member_to_rep(
            member_table, s_idx, sym_table.hkl_perm.to("cpu")
        ).squeeze(-1).numpy().astype(np.int64)
        aligned_tables.append(aligned)

    # 2. Pairwise residual misorientation for the agreement check.
    pair_misori = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # Use the chosen ops to express the pair's residual misorientation.
            # Cheap approximation: |misori(i,rep)| + |misori(j,rep)| upper-bounds it.
            pair_misori[i, j] = abs(member_misori[i]) + abs(member_misori[j])
            pair_misori[j, i] = pair_misori[i, j]

    # 3. Pairwise weights.
    _jacc, _agree, weights = _pairwise_weights(
        aligned_tables,
        misori_pair=pair_misori,
        ring_radii=ring_radii_per_hkl,
        pixel_size_um=pixel_size_um,
        pixel_tol=pixel_tol,
        alpha=merge_alpha,
    )

    # 4. Connected components on edges with weight > threshold (and self loops
    # so isolated members survive as their own grain).
    src_list = []
    dst_list = []
    for i in range(n):
        src_list.append(i)
        dst_list.append(i)
        for j in range(i + 1, n):
            if weights[i, j] > edge_weight_threshold:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    adj = coo_matrix(
        (np.ones(src.size, dtype=np.int8), (src, dst)),
        shape=(n, n),
    )
    n_components, comp_labels = connected_components(adj, directed=False, return_labels=True)

    # 5. Pack into GrainCandidate per component.
    grains: List[GrainCandidate] = []
    for c in range(n_components):
        idx = [i for i in range(n) if comp_labels[i] == c]
        if not idx:
            continue
        # Pick the rep for this sub-cluster: the rep_pos if it landed here,
        # else the member with smallest residual misorientation (mirrors min-IA).
        rep_local = None
        for i in idx:
            if member_positions[i] == rep_pos:
                rep_local = i
                break
        if rep_local is None:
            rep_local = idx[int(np.argmin(member_misori[idx]))]

        # Spot-floor (min_nr_spots): union of SpotIDs across members of this
        # sub-cluster. Drop sub-clusters with too few unique matched spots.
        union = set()
        for i in idx:
            union |= _spot_set(aligned_tables[i])
        if len(union) < min_nr_spots:
            continue

        grains.append(
            GrainCandidate(
                rep_pos=int(member_positions[rep_local]),
                member_positions=[int(member_positions[i]) for i in idx],
                member_sym_ops=[int(member_sym_ops[i]) for i in idx],
                aligned_spot_tables=[aligned_tables[i] for i in idx],
                member_misori_rad=member_misori[idx].copy(),
            )
        )
    return grains

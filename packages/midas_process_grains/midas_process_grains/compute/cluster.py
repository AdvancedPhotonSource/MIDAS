"""Phase 1 — orientation-only clustering.

Per the plan §4 Phase 1: gather seeds whose pairwise misorientation falls
below ``MisoriTol`` (default 0.25°). **No position gate** — per user
direction, position is unreliable and not a merge criterion.

Two execution paths:

1. **Naive** (``method="naive"``): blocked O(N²) pairwise misorientation.
   Used for tests / small smoke runs.

2. **Sym-bucketed** (``method="bucketed"``, default): each seed contributes
   **all 24 symmetry-equivalent Rodrigues representations** to the bucket
   hash, eliminating the FZ-boundary discontinuity that caused the v0.1
   bucketed code to miss virtually all cluster pairs at full-dataset
   scale. Candidate pairs are enumerated by walking the (sym-extended)
   bucket grid; the symmetry-aware ``misorientation_quat_batch`` then
   filters.

Vectorisation: all per-seed loops have been replaced with torch-batched
calls into ``midas_stress.orientation``. On the peakfit-hard dataset
(354 k alive seeds) this reduces Phase 1 wall-time from minutes-of-
Python-loops to seconds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from midas_stress.orientation import (
    fundamental_zone,
    make_symmetries,
    misorientation_quat_batch,
    orient_mat_to_quat,
)


__all__ = [
    "ClusterResult",
    "cluster_by_misorientation",
    "cluster_by_misorientation_from_orient_mats",
    "pairwise_misorientation",
    "bucket_candidate_pairs",
    "_quat_to_rodrigues",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    labels: np.ndarray              # (n_seeds,) int64 — -1 for non-alive
    n_clusters: int
    edges: np.ndarray               # (n_edges, 2) int64
    misori_edges: np.ndarray        # (n_edges,) float64 (radians)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_to_rodrigues(quats: np.ndarray) -> np.ndarray:
    """Convert ``(n, 4)`` unit quaternions (w, x, y, z) to Rodrigues vectors.

    ``r = (qx, qy, qz) / qw``, with sign chosen so that ``qw > 0`` (rotation
    in [0, 180°]).
    """
    qw = quats[:, 0]
    sgn = np.where(qw >= 0, 1.0, -1.0)
    qw_safe = sgn * np.clip(np.abs(qw), 1e-12, None)
    return quats[:, 1:] * sgn[:, None] / qw_safe[:, None]


def _quat_mul_broadcast(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Hamilton product q * s, broadcasting over leading dims.

    q : (..., 4) and s : (..., 4) — both real-first (w, x, y, z).
    Returns shape determined by broadcasting.
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sw, sx, sy, sz = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    rw = qw * sw - qx * sx - qy * sy - qz * sz
    rx = qw * sx + qx * sw + qy * sz - qz * sy
    ry = qw * sy - qx * sz + qy * sw + qz * sx
    rz = qw * sz + qx * sy - qy * sx + qz * sw
    return np.stack([rw, rx, ry, rz], axis=-1)


# ---------------------------------------------------------------------------
# Pairwise misorientation
# ---------------------------------------------------------------------------


def pairwise_misorientation(
    quats: np.ndarray,
    space_group: int,
    *,
    pair_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Symmetry-aware misorientation (radians) for selected (i, j) pairs.

    Always routes through the torch-vectorised
    ``_misorientation_quat_batch_torch`` path (passes torch tensors so the
    midas-stress dispatcher picks the batched torch impl). When the C
    extension isn't built, the numpy fallback is a Python per-pair loop
    that is 1000+× slower and fatal at 1.6M-pair scale.
    """
    n = quats.shape[0]
    if pair_indices is None:
        idx_i, idx_j = np.triu_indices(n, k=1)
        pair_indices = np.stack([idx_i, idx_j], axis=1)
    if pair_indices.size == 0:
        return np.empty((0,), dtype=np.float64)
    a = torch.from_numpy(np.ascontiguousarray(quats[pair_indices[:, 0]]))
    b = torch.from_numpy(np.ascontiguousarray(quats[pair_indices[:, 1]]))
    out = misorientation_quat_batch(a, b, space_group)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Symmetry-extended bucketing
# ---------------------------------------------------------------------------


def bucket_candidate_pairs(
    fz_quats: np.ndarray,
    misori_tol_rad: float,
    *,
    alive_idx: np.ndarray,
    space_group: int,
    safety_cells: int = 0,
) -> np.ndarray:
    """Return candidate (i, j) seed pairs whose 24 symmetry-equivalent
    quaternion representations fall in the same / adjacent 4-D bucket cells.

    Implementation note (v0.2): we bucket on the **4-D quaternion**
    coordinates with sign canonicalised so ``qw ≥ 0``, NOT on the
    Rodrigues vector. Reason: when a seed quaternion is multiplied by a
    symmetry op that includes a near-180° rotation, the resulting rep
    can have ``qw`` near 0; the Rodrigues vector ``(qx, qy, qz)/qw`` then
    diverges, and two physically-close orientations land in entirely
    different Rodrigues cells. The 4-D quaternion distance between two
    close orientations is ``2·sin(θ/2)`` regardless of ``qw`` magnitude,
    so a fixed 4-D cell size catches all near-pairs cleanly.

    Parameters
    ----------
    fz_quats : (n_seeds, 4)
        FZ-reduced quaternions (the FZ reduction itself is not strictly
        required by this routine — sym-extension handles it — but it
        keeps the rep set deterministic).
    misori_tol_rad : float
        Phase-1 misorientation threshold; sets cell size.
    alive_idx : (n_alive,)
        Global indices of alive seeds.
    space_group : int
        For ``make_symmetries``.
    safety_cells : int
        Half-width of the cell-neighbourhood search.

    Returns
    -------
    pair_indices : (m, 2) int64
        Sorted unique (i, j) with i < j; the caller still applies
        symmetry-aware misorientation to filter.
    """
    if alive_idx.size < 2:
        return np.empty((0, 2), dtype=np.int64)

    import time as _time
    _t0 = _time.time()

    # 1. Generate 24 symmetry-equivalent reps per alive seed.
    n_sym, sym_list = make_symmetries(space_group)
    sym_q = np.asarray(sym_list, dtype=np.float64)            # (n_sym, 4)
    q_alive = fz_quats[alive_idx]                              # (n_alive, 4)

    # Broadcast multiply: (n_alive, 1, 4) * (1, n_sym, 4) -> (n_alive, n_sym, 4)
    reps = _quat_mul_broadcast(q_alive[:, None, :], sym_q[None, :, :])
    reps_flat = reps.reshape(-1, 4)                            # (n_alive*n_sym, 4)
    seed_local = np.repeat(np.arange(alive_idx.size, dtype=np.int64), n_sym)
    print(f"[bucket] {alive_idx.size} seeds × {n_sym} reps = "
          f"{reps_flat.shape[0]:,} reps  [{_time.time()-_t0:.1f}s]", flush=True)

    # 2. Sign-canonicalise (qw ≥ 0) and bucket on the full 4-D quaternion.
    sgn = np.where(reps_flat[:, 0] >= 0, 1.0, -1.0)
    reps_flat = reps_flat * sgn[:, None]
    cell = 2.0 * math.sin(misori_tol_rad / 2.0)
    if cell <= 0:
        cell = 1e-9
    cell_idx = np.floor(reps_flat / cell).astype(np.int64)     # (n_alive*n_sym, 4)
    print(f"[bucket] cell idx built  [{_time.time()-_t0:.1f}s]", flush=True)

    # 3. Walk the (2*safety+1)^3 neighbourhood. For each offset, shift the
    #    cell indices, lex-sort, find runs, and emit cross pairs within
    #    each run.
    s = safety_cells
    # 4-D neighbourhood: (2*s+1)^4 offsets. With safety_cells=0 that's 1
    # (just same-cell match — relies on the 24 sym-rep extension + connected-
    # components transitivity to fill any cell-boundary gap). s=1 → 81,
    # s=2 → 625.
    offsets = [
        (dw, dx, dy, dz)
        for dw in range(-s, s + 1)
        for dx in range(-s, s + 1)
        for dy in range(-s, s + 1)
        for dz in range(-s, s + 1)
    ]

    pair_arrays: List[np.ndarray] = []
    n_offsets = len(offsets)
    for off_i, (dw, dx, dy, dz) in enumerate(offsets):
        if off_i % max(1, n_offsets // 10) == 0:
            print(f"[bucket] offset {off_i}/{n_offsets}  "
                  f"[{_time.time()-_t0:.1f}s]", flush=True)
        if (dw, dx, dy, dz) == (0, 0, 0, 0):
            shifted = cell_idx
        else:
            shifted = cell_idx + np.array([dw, dx, dy, dz], dtype=np.int64)

        order = np.lexsort(
            (shifted[:, 3], shifted[:, 2], shifted[:, 1], shifted[:, 0])
        )
        sorted_cells = shifted[order]
        sorted_seeds = seed_local[order]

        diff = np.any(np.diff(sorted_cells, axis=0) != 0, axis=1)
        breaks = np.concatenate([[0], np.flatnonzero(diff) + 1, [sorted_cells.shape[0]]])
        sizes = np.diff(breaks)

        # Vectorised pair emission: for every group with ≥ 2 distinct seeds,
        # generate all (i, j) seed pairs with i < j using np.triu_indices.
        # Pure-numpy hot loop, no Python-level pair iteration.
        big_groups = np.flatnonzero(sizes >= 2)
        for k in big_groups:
            lo, hi = int(breaks[k]), int(breaks[k + 1])
            members = np.unique(sorted_seeds[lo:hi])
            if members.size < 2:
                continue
            ii_idx, jj_idx = np.triu_indices(members.size, k=1)
            sub_pairs = np.stack([members[ii_idx], members[jj_idx]], axis=1)
            # Sort each row so smaller index is first.
            sub_pairs.sort(axis=1)
            pair_arrays.append(sub_pairs)
        if off_i == 0:
            tot_pairs_so_far = sum(p.shape[0] for p in pair_arrays)
            print(f"[bucket]   first offset done: {len(big_groups)} groups, "
                  f"{tot_pairs_so_far:,} pairs  [{_time.time()-_t0:.1f}s]",
                  flush=True)

    if not pair_arrays:
        return np.empty((0, 2), dtype=np.int64)

    # Concatenate, dedupe, sort.
    all_pairs = np.concatenate(pair_arrays, axis=0)
    print(f"[bucket] concat done: {all_pairs.shape[0]:,} pairs total "
          f"(pre-dedupe)  [{_time.time()-_t0:.1f}s]", flush=True)
    # Encode (a, b) as a single int64 for unique-ness, then decode.
    encoded = all_pairs[:, 0] * np.int64(2**31) + all_pairs[:, 1]
    unique_encoded = np.unique(encoded)
    print(f"[bucket] dedupe done: {unique_encoded.shape[0]:,} unique pairs  "
          f"[{_time.time()-_t0:.1f}s]", flush=True)
    a_out = unique_encoded // np.int64(2**31)
    b_out = unique_encoded % np.int64(2**31)
    # Map local indices to global seed indices.
    return np.stack([alive_idx[a_out], alive_idx[b_out]], axis=1)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def cluster_by_misorientation_from_orient_mats(
    orient_mats: np.ndarray,
    space_group: int,
    *,
    misori_tol_rad: float,
    alive_mask: Optional[np.ndarray] = None,
    method: str = "bucketed",
    block_size: int = 4096,
    safety_cells: int = 0,
    chunk: int = 200_000,
) -> ClusterResult:
    """Same as :func:`cluster_by_misorientation` but takes orientation matrices
    directly (vectorised OM→quat→FZ via midas-stress torch path).
    """
    n_seeds = orient_mats.shape[0]
    if alive_mask is None:
        alive_mask = np.ones(n_seeds, dtype=bool)
    else:
        alive_mask = np.asarray(alive_mask, dtype=bool)
    alive_idx = np.flatnonzero(alive_mask)

    # Vectorised OM → FZ-reduced quat via midas-stress torch path. One bulk
    # GPU/CPU op for the entire alive set.
    om_t = torch.from_numpy(
        np.ascontiguousarray(orient_mats[alive_idx], dtype=np.float64)
    )
    quats_alive_t = orient_mat_to_quat(om_t)                  # (n_alive, 4)
    fz_alive_t = fundamental_zone(quats_alive_t, space_group) # (n_alive, 4)
    fz_alive = fz_alive_t.detach().cpu().numpy()

    # Pad back to (n_seeds, 4) so downstream indexing by global pos works.
    fz_quats = np.zeros((n_seeds, 4), dtype=np.float64)
    fz_quats[:, 0] = 1.0          # identity for non-alive (irrelevant)
    fz_quats[alive_idx] = fz_alive

    return _cluster_from_fz_quats(
        fz_quats, space_group,
        misori_tol_rad=misori_tol_rad,
        alive_mask=alive_mask,
        method=method,
        block_size=block_size,
        safety_cells=safety_cells,
        chunk=chunk,
    )


def cluster_by_misorientation(
    quats: np.ndarray,
    space_group: int,
    *,
    misori_tol_rad: float,
    alive_mask: Optional[np.ndarray] = None,
    method: str = "bucketed",
    block_size: int = 4096,
    safety_cells: int = 0,
    chunk: int = 200_000,
) -> ClusterResult:
    """Cluster seed orientations by symmetry-aware misorientation.

    Parameters
    ----------
    quats : (n_seeds, 4)
        Raw (not necessarily FZ-reduced) quaternions.
    space_group : int
    misori_tol_rad : float
        Threshold (radians).
    alive_mask : (n_seeds,) bool, optional
        Non-alive seeds get label -1.
    method : {"bucketed", "naive"}
        Default ``"bucketed"`` uses the symmetry-extended Rodrigues bucket
        prefilter; ``"naive"`` does blocked O(N²) for testing.
    block_size : int
        Naive method block size.
    safety_cells : int
        Bucket-method neighbourhood half-width (default 1).
    chunk : int
        Max candidate pairs per ``misorientation_quat_batch`` call.
    """
    n_seeds = quats.shape[0]
    if alive_mask is None:
        alive_mask = np.ones(n_seeds, dtype=bool)
    else:
        alive_mask = np.asarray(alive_mask, dtype=bool)
        if alive_mask.shape != (n_seeds,):
            raise ValueError(f"alive_mask must be ({n_seeds},); got {alive_mask.shape}")

    # FZ-reduce alive seeds in one torch batch.
    alive_idx = np.flatnonzero(alive_mask)
    if alive_idx.size > 0:
        q_alive_t = torch.from_numpy(
            np.ascontiguousarray(quats[alive_idx], dtype=np.float64)
        )
        fz_alive_t = fundamental_zone(q_alive_t, space_group)
        fz_alive = fz_alive_t.detach().cpu().numpy()
    else:
        fz_alive = np.empty((0, 4), dtype=np.float64)

    fz_quats = np.zeros_like(quats)
    fz_quats[:, 0] = 1.0
    fz_quats[alive_idx] = fz_alive

    return _cluster_from_fz_quats(
        fz_quats, space_group,
        misori_tol_rad=misori_tol_rad,
        alive_mask=alive_mask,
        method=method,
        block_size=block_size,
        safety_cells=safety_cells,
        chunk=chunk,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _cluster_from_fz_quats(
    fz_quats: np.ndarray,
    space_group: int,
    *,
    misori_tol_rad: float,
    alive_mask: np.ndarray,
    method: str,
    block_size: int,
    safety_cells: int,
    chunk: int,
) -> ClusterResult:
    if method not in {"bucketed", "naive"}:
        raise ValueError(f"method must be 'bucketed' or 'naive'; got {method!r}")

    n_seeds = fz_quats.shape[0]
    alive_idx = np.flatnonzero(alive_mask)

    if method == "bucketed":
        candidate_pairs = bucket_candidate_pairs(
            fz_quats, misori_tol_rad,
            alive_idx=alive_idx, space_group=space_group,
            safety_cells=safety_cells,
        )
        edge_i_list, edge_j_list, edge_misori_list = [], [], []
        for start in range(0, candidate_pairs.shape[0], chunk):
            block = candidate_pairs[start:start + chunk]
            misori = pairwise_misorientation(
                fz_quats, space_group, pair_indices=block,
            )
            keep = misori < misori_tol_rad
            if keep.any():
                edge_i_list.append(block[keep, 0])
                edge_j_list.append(block[keep, 1])
                edge_misori_list.append(misori[keep])
        if edge_i_list:
            e_i = np.concatenate(edge_i_list).astype(np.int64)
            e_j = np.concatenate(edge_j_list).astype(np.int64)
            e_m = np.concatenate(edge_misori_list).astype(np.float64)
        else:
            e_i = np.empty(0, dtype=np.int64)
            e_j = np.empty(0, dtype=np.int64)
            e_m = np.empty(0, dtype=np.float64)
    else:                                                                  # naive
        edge_i, edge_j, edge_misori = [], [], []
        n_alive = alive_idx.size
        for a in range(0, n_alive, block_size):
            a_end = min(a + block_size, n_alive)
            block_a = alive_idx[a:a_end]
            for b in range(a, n_alive, block_size):
                b_end = min(b + block_size, n_alive)
                block_b = alive_idx[b:b_end]
                ii, jj = np.meshgrid(
                    np.arange(block_a.size), np.arange(block_b.size),
                    indexing="ij",
                )
                pair_local = np.stack([ii.ravel(), jj.ravel()], axis=1)
                if a == b:
                    keep_local = pair_local[:, 0] < pair_local[:, 1]
                    pair_local = pair_local[keep_local]
                if pair_local.size == 0:
                    continue
                pair_global = np.empty_like(pair_local)
                pair_global[:, 0] = block_a[pair_local[:, 0]]
                pair_global[:, 1] = block_b[pair_local[:, 1]]
                misori = pairwise_misorientation(
                    fz_quats, space_group, pair_indices=pair_global,
                )
                mask = misori < misori_tol_rad
                if mask.any():
                    edge_i.append(pair_global[mask, 0])
                    edge_j.append(pair_global[mask, 1])
                    edge_misori.append(misori[mask])
        if edge_i:
            e_i = np.concatenate(edge_i).astype(np.int64)
            e_j = np.concatenate(edge_j).astype(np.int64)
            e_m = np.concatenate(edge_misori).astype(np.float64)
        else:
            e_i = np.empty(0, dtype=np.int64)
            e_j = np.empty(0, dtype=np.int64)
            e_m = np.empty(0, dtype=np.float64)

    # Connected components — include self-edges so isolated alive seeds
    # become their own components.
    ones = np.ones(e_i.size + alive_idx.size, dtype=np.int8)
    src = np.concatenate([e_i, alive_idx])
    dst = np.concatenate([e_j, alive_idx])
    A = coo_matrix((ones, (src, dst)), shape=(n_seeds, n_seeds))
    n_components, labels = connected_components(
        A, directed=False, return_labels=True,
    )
    labels = labels.astype(np.int64)
    labels[~alive_mask] = -1

    used = sorted({int(x) for x in np.unique(labels) if x >= 0})
    if used:
        lut = np.full(int(max(used)) + 1, -1, dtype=np.int64)
        for new, old in enumerate(used):
            lut[old] = new
    else:
        lut = np.array([], dtype=np.int64)
    out_labels = np.full_like(labels, -1, dtype=np.int64)
    live = labels >= 0
    if live.any():
        out_labels[live] = lut[labels[live]]
    n_clusters = len(used)

    edges = np.stack([e_i, e_j], axis=1)
    return ClusterResult(
        labels=out_labels, n_clusters=n_clusters,
        edges=edges, misori_edges=e_m,
    )
